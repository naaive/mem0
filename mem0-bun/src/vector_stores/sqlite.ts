import { Database } from "bun:sqlite";
import { VectorStore } from "./base";
import { cosineSimilarity } from "../utils/hash";
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
 * - Vectors and payloads are JSON-encoded; cosine similarity is computed in
 *   TypeScript on the rows that pass scoped filters.
 * - Keyword search uses an FTS5 virtual table over `payload.data` with
 *   `porter unicode61` tokenization. The BM25 score from `bm25(fts)` is
 *   returned (negated so larger = more relevant).
 * - One database file per store instance. The Memory class derives a
 *   sibling path for the entity store automatically.
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
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS records (
        id TEXT PRIMARY KEY,
        vector_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        user_id TEXT,
        agent_id TEXT,
        run_id TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_records_user ON records(user_id);
      CREATE INDEX IF NOT EXISTS idx_records_agent ON records(agent_id);
      CREATE INDEX IF NOT EXISTS idx_records_run ON records(run_id);

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

  private deleteFts(id: string): void {
    this.db.prepare("DELETE FROM records_fts WHERE id = ?").run(id);
  }

  private extractScopedColumns(payload: Record<string, unknown>): {
    user_id: string | null;
    agent_id: string | null;
    run_id: string | null;
  } {
    return {
      user_id: payload.user_id ? String(payload.user_id) : null,
      agent_id: payload.agent_id ? String(payload.agent_id) : null,
      run_id: payload.run_id ? String(payload.run_id) : null,
    };
  }

  private rowToRecord(row: RowShape): {
    id: string;
    vector: number[];
    payload: Record<string, unknown>;
  } {
    return {
      id: row.id,
      vector: JSON.parse(row.vector_json) as number[],
      payload: JSON.parse(row.payload_json) as Record<string, unknown>,
    };
  }

  private buildScopedWhere(filters?: Record<string, unknown>): {
    sql: string;
    params: (string | number | null)[];
  } {
    const clauses: string[] = [];
    const params: (string | number | null)[] = [];
    if (filters) {
      for (const key of ["user_id", "agent_id", "run_id"] as const) {
        const v = filters[key];
        if (v !== undefined) {
          clauses.push(`${key} = ?`);
          params.push(typeof v === "number" ? v : String(v));
        }
      }
    }
    return {
      sql: clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "",
      params,
    };
  }

  private matchAdditionalFilters(
    payload: Record<string, unknown>,
    filters?: Record<string, unknown>,
  ): boolean {
    if (!filters) return true;
    for (const [k, v] of Object.entries(filters)) {
      if (v === undefined) continue;
      if (k === "user_id" || k === "agent_id" || k === "run_id") continue;
      if (payload[k] !== v) return false;
    }
    return true;
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
        const cols = this.extractScopedColumns(payload);
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
    const exists = this.db
      .prepare("SELECT 1 FROM records WHERE id = ?")
      .get(id);
    if (!exists) throw new Error(`Record ${id} not found`);
    const cols = this.extractScopedColumns(payload);
    this.db
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
    this.upsertFts(id, String(payload.data ?? ""));
  }

  async delete(id: string): Promise<void> {
    const tx = this.db.transaction(() => {
      this.db.prepare("DELETE FROM records WHERE id = ?").run(id);
      this.deleteFts(id);
    });
    tx();
  }

  async get(id: string): Promise<VectorRecord | null> {
    const row = this.db
      .prepare("SELECT * FROM records WHERE id = ?")
      .get(id) as RowShape | null;
    if (!row) return null;
    const rec = this.rowToRecord(row);
    return { id: rec.id, payload: rec.payload };
  }

  async list(
    filters?: Record<string, unknown>,
    limit = 100,
  ): Promise<VectorRecord[]> {
    const { sql, params } = this.buildScopedWhere(filters);
    // Pull a wider window so we can apply additional payload filters in TS.
    const rows = this.db
      .prepare(`SELECT * FROM records ${sql} LIMIT ?`)
      .all(...params, Math.max(limit * 4, limit)) as RowShape[];
    const out: VectorRecord[] = [];
    for (const row of rows) {
      const rec = this.rowToRecord(row);
      if (!this.matchAdditionalFilters(rec.payload, filters)) continue;
      out.push({ id: rec.id, payload: rec.payload });
      if (out.length >= limit) break;
    }
    return out;
  }

  async search(
    query: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]> {
    const { sql, params } = this.buildScopedWhere(filters);
    const rows = this.db
      .prepare(`SELECT * FROM records ${sql}`)
      .all(...params) as RowShape[];
    const scored: VectorRecord[] = [];
    for (const row of rows) {
      const rec = this.rowToRecord(row);
      if (!this.matchAdditionalFilters(rec.payload, filters)) continue;
      scored.push({
        id: rec.id,
        payload: rec.payload,
        score: cosineSimilarity(query, rec.vector),
      });
    }
    scored.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    return scored.slice(0, limit);
  }

  override async keywordSearch(
    queryTokens: string[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]> {
    if (queryTokens.length === 0) return [];
    // FTS5 expects an OR query with quoted tokens for safety.
    const matchExpr = queryTokens
      .map((t) => `"${t.replace(/"/g, '""')}"`)
      .join(" OR ");
    const { sql, params } = this.buildScopedWhere(filters);
    const fullSql = `
      SELECT records.*, bm25(records_fts) as fts_score
      FROM records_fts
      JOIN records ON records.id = records_fts.id
      ${sql ? sql.replace(/^WHERE/, "WHERE records_fts MATCH ? AND") : "WHERE records_fts MATCH ?"}
      ORDER BY fts_score
      LIMIT ?
    `;
    const allParams = sql ? [matchExpr, ...params, limit] : [matchExpr, limit];
    const rows = this.db
      .prepare(fullSql)
      .all(...allParams) as Array<RowShape & { fts_score: number }>;
    const out: VectorRecord[] = [];
    for (const row of rows) {
      const rec = this.rowToRecord(row);
      if (!this.matchAdditionalFilters(rec.payload, filters)) continue;
      // FTS5 returns bm25 in negative form (smaller = more relevant). Flip sign.
      out.push({
        id: rec.id,
        payload: rec.payload,
        score: -row.fts_score,
      });
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

  /** Visible for tests + Memory.close(): closes the underlying DB handle. */
  close(): void {
    this.db.close();
  }
}
