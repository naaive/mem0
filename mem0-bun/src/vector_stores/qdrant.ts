import { VectorStore } from "./base";
import type { VectorRecord, VectorStoreConfig } from "../types";

interface QdrantPoint {
  id: string;
  payload: Record<string, unknown>;
  score?: number;
}

/**
 * Qdrant client built on the REST API (no SDK dependency, fully testable
 * by stubbing global fetch).
 */
export class QdrantVectorStore extends VectorStore {
  private readonly baseURL: string;
  private readonly apiKey: string | undefined;
  private readonly collectionName: string;
  private readonly dimension: number;

  constructor(config: VectorStoreConfig = {}) {
    super(config);
    this.baseURL = (config.url ?? "http://localhost:6333").replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.collectionName = config.collectionName ?? "mem0_default";
    if (!config.dimension) {
      throw new Error("Qdrant vector store requires `dimension` in config");
    }
    this.dimension = config.dimension;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["api-key"] = this.apiKey;
    return h;
  }

  private async req(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<unknown> {
    const res = await fetch(`${this.baseURL}${path}`, {
      method,
      headers: this.headers(),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Qdrant ${method} ${path} ${res.status}: ${text}`);
    }
    return await res.json();
  }

  async initialize(): Promise<void> {
    const existing = (await this.req(
      "GET",
      `/collections/${this.collectionName}/exists`,
    )) as { result?: { exists?: boolean } };
    if (existing.result?.exists) return;
    await this.req("PUT", `/collections/${this.collectionName}`, {
      vectors: { size: this.dimension, distance: "Cosine" },
    });
  }

  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, unknown>[],
  ): Promise<void> {
    const points = vectors.map((vector, i) => ({
      id: ids[i]!,
      vector,
      payload: payloads[i]!,
    }));
    await this.req(
      "PUT",
      `/collections/${this.collectionName}/points?wait=true`,
      { points },
    );
  }

  private buildFilter(
    filters?: Record<string, unknown>,
  ): Record<string, unknown> | undefined {
    if (!filters) return undefined;
    const must: Array<Record<string, unknown>> = [];
    for (const [key, value] of Object.entries(filters)) {
      if (value === undefined) continue;
      must.push({ key, match: { value } });
    }
    return must.length > 0 ? { must } : undefined;
  }

  async search(
    query: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]> {
    const body: Record<string, unknown> = {
      vector: query,
      limit,
      with_payload: true,
    };
    const filter = this.buildFilter(filters);
    if (filter) body.filter = filter;
    const data = (await this.req(
      "POST",
      `/collections/${this.collectionName}/points/search`,
      body,
    )) as { result?: QdrantPoint[] };
    return (data.result ?? []).map((p) => ({
      id: String(p.id),
      payload: p.payload ?? {},
      score: p.score,
    }));
  }

  async get(id: string): Promise<VectorRecord | null> {
    try {
      const data = (await this.req(
        "GET",
        `/collections/${this.collectionName}/points/${id}`,
      )) as { result?: QdrantPoint };
      if (!data.result) return null;
      return { id: String(data.result.id), payload: data.result.payload ?? {} };
    } catch {
      return null;
    }
  }

  async update(
    id: string,
    vector: number[],
    payload: Record<string, unknown>,
  ): Promise<void> {
    await this.insert([vector], [id], [payload]);
  }

  async delete(id: string): Promise<void> {
    await this.req(
      "POST",
      `/collections/${this.collectionName}/points/delete?wait=true`,
      { points: [id] },
    );
  }

  async list(
    filters?: Record<string, unknown>,
    limit = 100,
  ): Promise<VectorRecord[]> {
    const body: Record<string, unknown> = {
      limit,
      with_payload: true,
    };
    const filter = this.buildFilter(filters);
    if (filter) body.filter = filter;
    const data = (await this.req(
      "POST",
      `/collections/${this.collectionName}/points/scroll`,
      body,
    )) as { result?: { points?: QdrantPoint[] } };
    return (data.result?.points ?? []).map((p) => ({
      id: String(p.id),
      payload: p.payload ?? {},
    }));
  }

  async deleteCollection(): Promise<void> {
    await this.req("DELETE", `/collections/${this.collectionName}`);
  }
}
