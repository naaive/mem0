import type { VectorRecord, VectorStoreConfig } from "../types";

export abstract class VectorStore {
  protected readonly config: VectorStoreConfig;

  constructor(config: VectorStoreConfig = {}) {
    this.config = config;
  }

  abstract initialize(): Promise<void>;
  abstract insert(
    vectors: number[][],
    ids: string[],
    payloads: Record<string, unknown>[],
  ): Promise<void>;
  abstract search(
    query: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]>;
  abstract get(id: string): Promise<VectorRecord | null>;
  abstract update(
    id: string,
    vector: number[],
    payload: Record<string, unknown>,
  ): Promise<void>;
  abstract delete(id: string): Promise<void>;
  abstract list(
    filters?: Record<string, unknown>,
    limit?: number,
  ): Promise<VectorRecord[]>;
  abstract deleteCollection(): Promise<void>;

  /**
   * Optional keyword (BM25) search. Implementations that don't support
   * full-text scoring should leave this undefined; the Memory class falls
   * back to semantic-only retrieval.
   */
  keywordSearch?(
    queryTokens: string[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorRecord[]>;

  /** Release backing resources (DB handles, sockets). No-op by default. */
  close(): void {
    /* no-op */
  }
}
