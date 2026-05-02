import { cosineSimilarity } from "../utils/hash";
import {
  GraphStore,
  type GraphSearchOptions,
  type Triple,
  type TripleMatch,
} from "./base";

interface EmbeddedTriple extends Triple {
  subjectVec: number[];
  objectVec: number[];
}

/**
 * Local-first in-process graph store. Triples are kept in a list and indexed
 * by entity name for fast hop traversal. Embedding similarity is used for
 * fuzzy entity matching during retrieval.
 *
 * Suitable for tests, prototyping, and small-corpus eval. For multi-million
 * triple corpora users should plug in a Kuzu/Neo4j adapter that implements
 * the same `GraphStore` interface.
 */
export class InMemoryGraphStore extends GraphStore {
  private triples: EmbeddedTriple[] = [];
  /** Index: lowercased entity name → triple indices where it appears. */
  private entityIndex = new Map<string, Set<number>>();
  /** Cache of embeddings for entity strings to avoid re-embedding. */
  private embeddingCache = new Map<string, number[]>();

  constructor() {
    super();
  }

  async initialize(): Promise<void> {
    // Nothing to do; in-memory init is implicit.
  }

  private indexEntity(name: string, idx: number): void {
    const key = name.toLowerCase().trim();
    if (!key) return;
    let set = this.entityIndex.get(key);
    if (!set) {
      set = new Set();
      this.entityIndex.set(key, set);
    }
    set.add(idx);
  }

  private removeEntity(name: string, idx: number): void {
    const key = name.toLowerCase().trim();
    const set = this.entityIndex.get(key);
    if (!set) return;
    set.delete(idx);
    if (set.size === 0) this.entityIndex.delete(key);
  }

  private filtersMatch(
    triple: Triple,
    filters: Record<string, unknown>,
  ): boolean {
    for (const [k, v] of Object.entries(filters)) {
      if (v === undefined) continue;
      if (triple.filters[k] !== v) return false;
    }
    return true;
  }

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

  async addTriples(
    triples: Triple[],
    embed: (text: string) => Promise<number[]>,
  ): Promise<void> {
    for (const t of triples) {
      const subjectVec = await this.getEmbedding(t.subject, embed);
      const objectVec = await this.getEmbedding(t.object, embed);
      const idx = this.triples.length;
      this.triples.push({ ...t, subjectVec, objectVec });
      this.indexEntity(t.subject, idx);
      this.indexEntity(t.object, idx);
    }
  }

  async searchByEntities(
    entityEmbeddings: Array<{ text: string; vector: number[] }>,
    options: GraphSearchOptions,
  ): Promise<TripleMatch[]> {
    if (this.triples.length === 0 || entityEmbeddings.length === 0) return [];
    const limit = options.limit ?? 50;
    const hops = Math.max(1, options.hops ?? 1);

    // Step 1 — find candidate triples whose subject or object is similar
    // to one of the query entities (≥ 0.7 cosine).
    const seedScores = new Map<number, number>(); // triple idx → best score
    for (const ent of entityEmbeddings) {
      for (let i = 0; i < this.triples.length; i++) {
        const tr = this.triples[i]!;
        if (!this.filtersMatch(tr, options.filters)) continue;
        const subSim = cosineSimilarity(ent.vector, tr.subjectVec);
        const objSim = cosineSimilarity(ent.vector, tr.objectVec);
        const sim = Math.max(subSim, objSim);
        if (sim < 0.7) continue;
        const prev = seedScores.get(i) ?? 0;
        if (sim > prev) seedScores.set(i, sim);
      }
    }
    if (seedScores.size === 0) return [];

    // Step 2 — expand by `hops`. At each hop we collect every triple that
    // shares an entity with already-visited triples.
    const visited = new Map<number, number>(seedScores);
    let frontier = new Set<number>(seedScores.keys());
    for (let h = 1; h < hops; h++) {
      const next = new Set<number>();
      for (const idx of frontier) {
        const tr = this.triples[idx]!;
        for (const name of [tr.subject, tr.object]) {
          const peers = this.entityIndex.get(name.toLowerCase().trim()) ?? [];
          for (const p of peers) {
            if (visited.has(p)) continue;
            const candidate = this.triples[p]!;
            if (!this.filtersMatch(candidate, options.filters)) continue;
            // Hop-attenuated score
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
      const tr = this.triples[idx]!;
      matches.push({
        subject: tr.subject,
        relation: tr.relation,
        object: tr.object,
        filters: tr.filters,
        memoryId: tr.memoryId,
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
    // Compact: rebuild triples list excluding matches; rebuild indexes.
    const kept: EmbeddedTriple[] = [];
    for (const tr of this.triples) {
      if (
        tr.memoryId === memoryId &&
        this.filtersMatch(tr, filters)
      ) {
        continue;
      }
      kept.push(tr);
    }
    this.triples = kept;
    this.entityIndex.clear();
    for (let i = 0; i < this.triples.length; i++) {
      const tr = this.triples[i]!;
      this.indexEntity(tr.subject, i);
      this.indexEntity(tr.object, i);
    }
  }

  async reset(): Promise<void> {
    this.triples = [];
    this.entityIndex.clear();
    this.embeddingCache.clear();
  }

  /** Visible for tests: returns the number of stored triples. */
  size(): number {
    return this.triples.length;
  }

  /** Visible for tests: helper to remove a specific entity from the index. */
  unindexEntity(name: string, idx: number): void {
    this.removeEntity(name, idx);
  }
}
