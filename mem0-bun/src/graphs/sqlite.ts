import { Database } from "bun:sqlite";
import { cosineSimilarity } from "../utils/hash";
import {
  GraphStore,
  type GraphSearchOptions,
  type Triple,
  type TripleMatch,
} from "./base";

interface TripleRow {
  id: number;
  subject: string;
  relation: string;
  object: string;
  subject_vec_json: string;
  object_vec_json: string;
  filters_json: string;
  memory_id: string | null;
  user_id: string | null;
  agent_id: string | null;
  run_id: string | null;
}

export interface SqliteGraphStoreConfig {
  path?: string;
}

/**
 * Disk-backed embedded graph store on top of `bun:sqlite`.
 *
 * Triples are kept in a single table with vectors JSON-encoded; entity
 * lookup uses indexed `subject` / `object` columns. Fuzzy entity matching
 * during retrieval is computed in TypeScript over the rows that pass the
 * scoped filter (so the cost stays linear in scoped triples, not in the
 * whole DB).
 */
export class SqliteGraphStore extends GraphStore {
  private readonly db: Database;
  /** Cache of entity-text → vector across the lifetime of this instance. */
  private readonly embeddingCache = new Map<string, number[]>();

  constructor(config: SqliteGraphStoreConfig = {}) {
    super();
    this.db = new Database(config.path ?? ":memory:");
    this.db.exec("PRAGMA journal_mode = WAL");
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS triples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT NOT NULL,
        relation TEXT NOT NULL,
        object TEXT NOT NULL,
        subject_vec_json TEXT NOT NULL,
        object_vec_json TEXT NOT NULL,
        filters_json TEXT NOT NULL,
        memory_id TEXT,
        user_id TEXT,
        agent_id TEXT,
        run_id TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
      CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
      CREATE INDEX IF NOT EXISTS idx_triples_memory ON triples(memory_id);
      CREATE INDEX IF NOT EXISTS idx_triples_user ON triples(user_id);
      CREATE INDEX IF NOT EXISTS idx_triples_agent ON triples(agent_id);
      CREATE INDEX IF NOT EXISTS idx_triples_run ON triples(run_id);
    `);
  }

  async initialize(): Promise<void> {
    // Schema setup is synchronous in the constructor.
  }

  // ---------------- helpers ----------------

  private async getEmbedding(
    text: string,
    embed: (text: string) => Promise<number[]>,
  ): Promise<number[]> {
    const key = text.toLowerCase().trim();
    let v = this.embeddingCache.get(key);
    if (!v) {
      v = await embed(text);
      this.embeddingCache.set(key, v);
    }
    return v;
  }

  private buildScopedWhere(filters: Record<string, unknown>): {
    sql: string;
    params: (string | number | null)[];
  } {
    const clauses: string[] = [];
    const params: (string | number | null)[] = [];
    for (const key of ["user_id", "agent_id", "run_id"] as const) {
      const v = filters[key];
      if (v !== undefined) {
        clauses.push(`${key} = ?`);
        params.push(typeof v === "number" ? v : String(v));
      }
    }
    return {
      sql: clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "",
      params,
    };
  }

  private matchAdditionalFilters(
    storedFilters: Record<string, unknown>,
    queryFilters: Record<string, unknown>,
  ): boolean {
    for (const [k, v] of Object.entries(queryFilters)) {
      if (v === undefined) continue;
      if (k === "user_id" || k === "agent_id" || k === "run_id") continue;
      if (storedFilters[k] !== v) return false;
    }
    return true;
  }

  private rowToTriple(row: TripleRow): {
    triple: Triple;
    subjectVec: number[];
    objectVec: number[];
  } {
    const filters = JSON.parse(row.filters_json) as Record<string, unknown>;
    return {
      triple: {
        subject: row.subject,
        relation: row.relation,
        object: row.object,
        filters,
        memoryId: row.memory_id ?? undefined,
      },
      subjectVec: JSON.parse(row.subject_vec_json) as number[],
      objectVec: JSON.parse(row.object_vec_json) as number[],
    };
  }

  // ---------------- API ----------------

  async addTriples(
    triples: Triple[],
    embed: (text: string) => Promise<number[]>,
  ): Promise<void> {
    const insert = this.db.prepare(`
      INSERT INTO triples (
        subject, relation, object,
        subject_vec_json, object_vec_json,
        filters_json, memory_id,
        user_id, agent_id, run_id
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    for (const t of triples) {
      const subjectVec = await this.getEmbedding(t.subject, embed);
      const objectVec = await this.getEmbedding(t.object, embed);
      insert.run(
        t.subject,
        t.relation,
        t.object,
        JSON.stringify(subjectVec),
        JSON.stringify(objectVec),
        JSON.stringify(t.filters),
        t.memoryId ?? null,
        t.filters.user_id ? String(t.filters.user_id) : null,
        t.filters.agent_id ? String(t.filters.agent_id) : null,
        t.filters.run_id ? String(t.filters.run_id) : null,
      );
    }
  }

  async searchByEntities(
    entityEmbeddings: Array<{ text: string; vector: number[] }>,
    options: GraphSearchOptions,
  ): Promise<TripleMatch[]> {
    if (entityEmbeddings.length === 0) return [];
    const limit = options.limit ?? 50;
    const hops = Math.max(1, options.hops ?? 1);
    const { sql, params } = this.buildScopedWhere(options.filters);
    const rows = this.db
      .prepare(`SELECT * FROM triples ${sql}`)
      .all(...params) as TripleRow[];
    if (rows.length === 0) return [];

    // Decode rows once; we'll iterate them for both seed and hop expansion.
    const decoded = rows.map((row) => ({
      ...this.rowToTriple(row),
      idx: row.id,
    }));

    // Index entity name → row indices for fast hop expansion.
    const entityIndex = new Map<string, number[]>();
    for (let i = 0; i < decoded.length; i++) {
      const t = decoded[i]!.triple;
      for (const name of [t.subject, t.object]) {
        const key = name.toLowerCase().trim();
        if (!key) continue;
        let arr = entityIndex.get(key);
        if (!arr) {
          arr = [];
          entityIndex.set(key, arr);
        }
        arr.push(i);
      }
    }

    // Step 1 — seed by similarity.
    const visited = new Map<number, number>(); // index → best score
    for (const ent of entityEmbeddings) {
      for (let i = 0; i < decoded.length; i++) {
        const d = decoded[i]!;
        if (
          !this.matchAdditionalFilters(d.triple.filters, options.filters)
        ) {
          continue;
        }
        const subSim = cosineSimilarity(ent.vector, d.subjectVec);
        const objSim = cosineSimilarity(ent.vector, d.objectVec);
        const sim = Math.max(subSim, objSim);
        if (sim < 0.7) continue;
        const prev = visited.get(i) ?? 0;
        if (sim > prev) visited.set(i, sim);
      }
    }
    if (visited.size === 0) return [];

    // Step 2 — k-hop expansion.
    let frontier = new Set<number>(visited.keys());
    for (let h = 1; h < hops; h++) {
      const next = new Set<number>();
      for (const idx of frontier) {
        const d = decoded[idx]!;
        for (const name of [d.triple.subject, d.triple.object]) {
          const peers =
            entityIndex.get(name.toLowerCase().trim()) ?? [];
          for (const p of peers) {
            if (visited.has(p)) continue;
            const peer = decoded[p]!;
            if (
              !this.matchAdditionalFilters(peer.triple.filters, options.filters)
            ) {
              continue;
            }
            visited.set(p, (visited.get(idx) ?? 0) * 0.7);
            next.add(p);
          }
        }
      }
      frontier = next;
      if (frontier.size === 0) break;
    }

    const matches: TripleMatch[] = [];
    for (const [idx, score] of visited) {
      const d = decoded[idx]!;
      matches.push({
        subject: d.triple.subject,
        relation: d.triple.relation,
        object: d.triple.object,
        filters: d.triple.filters,
        memoryId: d.triple.memoryId,
        score,
      });
    }
    matches.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    return matches.slice(0, limit);
  }

  async deleteByMemoryId(
    memoryId: string,
    filters: Record<string, unknown>,
  ): Promise<void> {
    const { sql, params } = this.buildScopedWhere(filters);
    const where = sql ? `${sql} AND memory_id = ?` : "WHERE memory_id = ?";
    this.db
      .prepare(`DELETE FROM triples ${where}`)
      .run(...params, memoryId);
  }

  async reset(): Promise<void> {
    this.db.exec("DELETE FROM triples");
    this.embeddingCache.clear();
  }

  /** Visible for tests + Memory.close(): closes the underlying DB handle. */
  close(): void {
    this.db.close();
  }

  /** Visible for tests: total triple count. */
  size(): number {
    const row = this.db.prepare("SELECT COUNT(*) as c FROM triples").get() as
      | { c: number }
      | null;
    return row?.c ?? 0;
  }
}
