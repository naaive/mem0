import { Database } from "bun:sqlite";
import { VectorStore } from "./base";
import { cosineSimilarity } from "../utils/hash";
import {
  SCOPED_KEYS,
  buildScopedWhere,
  extractScopedColumns,
  hasExtraFilters,
  matchPayloadFilters,
} from "../utils/sqlite_filters";
import type { VectorRecord, VectorStoreConfig } from "../types";

interface RowShape {
  id: string;
  vector_json: string;
  payload_json: string;
  user_id: string | null;
  agent_id: string | null;
  run_id: string | null;
}

/**
 * Disk-backed embedded vector store on top of `bun:sqlite`.
 *
 * Vectors and payloads are JSON-encoded; cosine similarity is computed in
 * TypeScript on rows that pass scoped filters. Keyword search uses an
 * FTS5 virtual table over `payload.data` with `porter unicode61`
 * tokenization and the built-in `bm25()` ranker.
 */
export class SqliteVectorStore extends VectorStore {
  private readonly db: Database;
  private readonly collectionName: string;

  constructor(config: VectorStoreConfig = {}) {
    super(config);
    this.collectionName = config.collectionName ?? "mem0_default";
    const path = (config.path as string | undefined) ?? ":memory:";
    this.db = new Database(path);
    this.db.exec("PRAGMA journal_mode = WAL");
    const scopedColumns = SCOPED_KEYS.map((k) => `${k} TEXT`).join(", ");
    const scopedIndexes = SCOPED_KEYS.map(
      (k) => `CREATE INDEX IF NOT EXISTS idx_records_${k} ON records(${k})`,
    ).join(";\n      ");
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS records (
        id TEXT PRIMARY KEY,
        vector_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        ${scopedColumns}
      );
      ${scopedIndexes};

      CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
        id UNINDEXED,
        data,
        tokenize='porter unicode61'
      );
    `);
  }

  async initialize(): Promise<void> {
    // Schema is created in the constructor; nothing async to do.
  }

  // ---------------- helpers ----------------

  private upsertFts(id: string, data: string): void {
    this.db.prepare("DELETE FROM records_fts WHERE id = ?").run(id);
    this.db
      .prepare("INSERT INTO records_fts (id, data) VALUES (?, ?)")
      .run(id, data);
  }

  // ---------------- CRUD ----------------

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, unknown>[],
  ): Promise<void> {
    if (vectors.length !== ids.length || vectors.length !== payloads.length) {
      throw new Error("vectors, ids, and payloads must have equal length");
    }
    const stmt = this.db.prepare(
      "INSERT OR REPLACE INTO records (id, vector_json, payload_json, user_id, agent_id, run_id) VALUES (?, ?, ?, ?, ?, ?)",
    );
    const tx = this.db.transaction(() => {
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i]!;
        const payload = payloads[i]!;
        const cols = extractScopedColumns(payload);
        stmt.run(
          id,
          JSON.stringify(vectors[i]),
          JSON.stringify(payload),
          cols.user_id,
          cols.agent_id,
          cols.run_id,
        );
        this.upsertFts(id, String(payload.data ?? ""));
      }
    });
    tx();
  }

  async update(
    id: string,
    vector: number[],
    payload: Record<string, unknown>,
  ): Promise<void> {
    const cols = extractScopedColumns(payload);
    const result = this.db
      .prepare(
        "UPDATE records SET vector_json = ?, payload_json = ?, user_id = ?, agent_id = ?, run_id = ? WHERE id = ?",
      )
      .run(
        JSON.stringify(vector),
        JSON.stringify(payload),
        cols.user_id,
        cols.agent_id,
        cols.run_id,
        id,
      );
    if (result.changes === 0) throw new Error(`Record ${id} not found`);
    this.upsertFts(id, String(payload.data ?? ""));
  }

  async delete(id: string): Promise<void> {
    const tx = this.db.transaction(() => {
      this.db.prepare("DELETE FROM records WHERE id = ?").run(id);
      this.db.prepare("DELETE FROM records_fts WHERE id = ?").run(id);
    });
    tx();
  }

  async get(id: string): Promise<VectorRecord | null> {
    const row = this.db
      .prepare("SELECT * FROM records WHERE id = ?")
      .get(id) as RowShape | null;
    if (!row) return null;
    return { id: row.id, payload: JSON.parse(row.payload_json) };
  }

  async list(
    filters?: Record<string, unknown>,
    limit = 100,
  ): Promise<VectorRecord[]> {
    const { sql, params } = buildScopedWhere(filters);
    // Over-fetch only when extra payload filters require post-filtering.
    const fetch = hasExtraFilters(filters) ? Math.max(limit * 4, limit) : limit;
    const rows = this.db
      .prepare(
        `SELECT id, payload_json FROM records ${sql} ORDER BY rowid LIMIT ?`,
      )
      .all(...params, fetch) as Pick<RowShape, "id" | "payload_json">[];
    const out: VectorRecord[] = [];
    for (const row of rows) {
      const payload = JSON.parse(row.payload_json) as Record<string, unknown>;
      if (!matchPayloadFilters(payload, filters)) continue;
      out.push({ id: row.id, payload });
      if (out.length >= limit) break;
    }
    return out;
  }

  async search(
    query: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]> {
    const { sql, params } = buildScopedWhere(filters);
    const rows = this.db
      .prepare(`SELECT id, vector_json, payload_json FROM records ${sql}`)
      .all(...params) as Pick<
      RowShape,
      "id" | "vector_json" | "payload_json"
    >[];
    // Pass 1: parse vectors, score, keep payload_json string (lazy).
    const scored: Array<{ id: string; payload_json: string; score: number }> =
      [];
    for (const row of rows) {
      const score = cosineSimilarity(query, JSON.parse(row.vector_json));
      scored.push({ id: row.id, payload_json: row.payload_json, score });
    }
    scored.sort((a, b) => b.score - a.score);
    // Pass 2: parse payloads only for survivors, applying extra filters.
    const out: VectorRecord[] = [];
    for (const s of scored) {
      const payload = JSON.parse(s.payload_json) as Record<string, unknown>;
      if (!matchPayloadFilters(payload, filters)) continue;
      out.push({ id: s.id, payload, score: s.score });
      if (out.length >= limit) break;
    }
    return out;
  }

  override async keywordSearch(
    queryTokens: string[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]> {
    if (queryTokens.length === 0) return [];
    const matchExpr = queryTokens
      .map((t) => `"${t.replace(/"/g, '""')}"`)
      .join(" OR ");
    const { sql: scopedSql, params: scopedParams } = buildScopedWhere(filters);
    // Build the WHERE programmatically: FTS MATCH first, then scoped clauses.
    const clauses = ["records_fts MATCH ?"];
    if (scopedSql) clauses.push(scopedSql.replace(/^WHERE /, ""));
    const fullSql = `
      SELECT records.id, records.payload_json, bm25(records_fts) as fts_score
      FROM records_fts
      JOIN records ON records.id = records_fts.id
      WHERE ${clauses.join(" AND ")}
      ORDER BY fts_score
      LIMIT ?
    `;
    const rows = this.db
      .prepare(fullSql)
      .all(matchExpr, ...scopedParams, limit) as Array<{
      id: string;
      payload_json: string;
      fts_score: number;
    }>;
    const out: VectorRecord[] = [];
    for (const row of rows) {
      const payload = JSON.parse(row.payload_json) as Record<string, unknown>;
      if (!matchPayloadFilters(payload, filters)) continue;
      // FTS5 returns bm25 in negative form (smaller = more relevant). Flip sign.
      out.push({ id: row.id, payload, score: -row.fts_score });
    }
    return out;
  }

  async deleteCollection(): Promise<void> {
    const tx = this.db.transaction(() => {
      this.db.exec("DELETE FROM records");
      this.db.exec("DELETE FROM records_fts");
    });
    tx();
  }

  override close(): void {
    this.db.close();
  }
}
