import { VectorStore } from "./base";
import { cosineSimilarity } from "../utils/hash";
import {
  buildCorpusStats,
  scoreBm25,
  type CorpusStats,
} from "../utils/bm25";
import type { VectorRecord, VectorStoreConfig } from "../types";

interface InternalRecord {
  id: string;
  vector: number[];
  payload: Record<string, unknown>;
  /** Lemmatized tokens for BM25 (read from payload.textLemmatized). */
  tokens: string[];
}

/**
 * In-process vector store. Local-first default: all data lives in memory,
 * filtered linearly. Supports both semantic search (cosine) and a BM25
 * keyword search over `payload.textLemmatized` (string[] of stems).
 */
export class InMemoryVectorStore extends VectorStore {
  private readonly collections = new Map<string, Map<string, InternalRecord>>();
  private readonly collectionName: string;
  // Cached corpus stats per collection — invalidated on mutation.
  private statsCache = new Map<string, CorpusStats>();

  constructor(config: VectorStoreConfig = {}) {
    super(config);
    this.collectionName = config.collectionName ?? "mem0_default";
  }

  private store(): Map<string, InternalRecord> {
    let col = this.collections.get(this.collectionName);
    if (!col) {
      col = new Map();
      this.collections.set(this.collectionName, col);
    }
    return col;
  }

  private invalidateStats(): void {
    this.statsCache.delete(this.collectionName);
  }

  private getStats(): CorpusStats {
    let stats = this.statsCache.get(this.collectionName);
    if (!stats) {
      const docs: string[][] = [];
      for (const rec of this.store().values()) docs.push(rec.tokens);
      stats = buildCorpusStats(docs);
      this.statsCache.set(this.collectionName, stats);
    }
    return stats;
  }

  async initialize(): Promise<void> {
    this.store(); // ensure collection exists
  }

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, unknown>[],
  ): Promise<void> {
    if (vectors.length !== ids.length || vectors.length !== payloads.length) {
      throw new Error("vectors, ids, and payloads must have equal length");
    }
    const col = this.store();
    for (let i = 0; i < ids.length; i++) {
      const payload = payloads[i]!;
      const tokens = Array.isArray(payload.textLemmatized)
        ? (payload.textLemmatized as string[])
        : [];
      col.set(ids[i]!, {
        id: ids[i]!,
        vector: vectors[i]!,
        payload,
        tokens,
      });
    }
    this.invalidateStats();
  }

  private matchFilters(
    payload: Record<string, unknown>,
    filters?: Record<string, unknown>,
  ): boolean {
    if (!filters) return true;
    for (const [key, value] of Object.entries(filters)) {
      if (value === undefined) continue;
      if (payload[key] !== value) return false;
    }
    return true;
  }

  async search(
    query: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]> {
    const results: VectorRecord[] = [];
    for (const rec of this.store().values()) {
      if (!this.matchFilters(rec.payload, filters)) continue;
      results.push({
        id: rec.id,
        payload: rec.payload,
        score: cosineSimilarity(query, rec.vector),
      });
    }
    results.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    return results.slice(0, limit);
  }

  override async keywordSearch(
    queryTokens: string[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]> {
    const stats = this.getStats();
    const results: VectorRecord[] = [];
    for (const rec of this.store().values()) {
      if (!this.matchFilters(rec.payload, filters)) continue;
      const score = scoreBm25(queryTokens, rec.tokens, stats);
      if (score === 0) continue;
      results.push({ id: rec.id, payload: rec.payload, score });
    }
    results.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    return results.slice(0, limit);
  }

  async get(id: string): Promise<VectorRecord | null> {
    const rec = this.store().get(id);
    if (!rec) return null;
    return { id: rec.id, payload: rec.payload };
  }

  async update(
    id: string,
    vector: number[],
    payload: Record<string, unknown>,
  ): Promise<void> {
    const col = this.store();
    if (!col.has(id)) {
      throw new Error(`Record ${id} not found`);
    }
    const tokens = Array.isArray(payload.textLemmatized)
      ? (payload.textLemmatized as string[])
      : [];
    col.set(id, { id, vector, payload, tokens });
    this.invalidateStats();
  }

  async delete(id: string): Promise<void> {
    const removed = this.store().delete(id);
    if (removed) this.invalidateStats();
  }

  async list(
    filters?: Record<string, unknown>,
    limit = 100,
  ): Promise<VectorRecord[]> {
    const results: VectorRecord[] = [];
    for (const rec of this.store().values()) {
      if (!this.matchFilters(rec.payload, filters)) continue;
      results.push({ id: rec.id, payload: rec.payload });
      if (results.length >= limit) break;
    }
    return results;
  }

  async deleteCollection(): Promise<void> {
    this.collections.delete(this.collectionName);
    this.statsCache.delete(this.collectionName);
  }
}
