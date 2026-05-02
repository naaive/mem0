import { VectorStore } from "./base";
import { cosineSimilarity } from "../utils/hash";
import type { VectorRecord, VectorStoreConfig } from "../types";

interface InternalRecord {
  id: string;
  vector: number[];
  payload: Record<string, unknown>;
}

/**
 * In-process vector store. Local-first default: all data lives in memory,
 * filtered linearly. Suitable for tests, prototyping, and small datasets.
 */
export class InMemoryVectorStore extends VectorStore {
  private readonly collections = new Map<string, Map<string, InternalRecord>>();
  private readonly collectionName: string;

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
      col.set(ids[i]!, {
        id: ids[i]!,
        vector: vectors[i]!,
        payload: payloads[i]!,
      });
    }
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
    col.set(id, { id, vector, payload });
  }

  async delete(id: string): Promise<void> {
    this.store().delete(id);
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
  }
}
