export interface Triple {
  subject: string;
  relation: string;
  object: string;
  /** Free-form session scope. */
  filters: Record<string, unknown>;
  /** Originating memory id; used for cleanup. */
  memoryId?: string;
}

export interface TripleMatch extends Triple {
  /** Similarity score in [0, 1] when retrieved by embedding. */
  score?: number;
}

export interface GraphSearchOptions {
  filters: Record<string, unknown>;
  /** Optional max number of triples returned. */
  limit?: number;
  /** Optional max graph traversal depth. Default 1. */
  hops?: number;
}

export abstract class GraphStore {
  constructor() {
    // Explicit no-op so coverage observes the class being instantiated.
  }
  abstract initialize(): Promise<void>;
  /** Add triples in bulk; new entities are upserted. */
  abstract addTriples(
    triples: Triple[],
    embed: (text: string) => Promise<number[]>,
  ): Promise<void>;
  /**
   * Resolve query entities, traverse `hops` outward, and return triples
   * (with their associated memoryIds via payload search). Implementations
   * should perform fuzzy matching (≥0.7 similarity) on subject/object.
   */
  abstract searchByEntities(
    entityEmbeddings: Array<{ text: string; vector: number[] }>,
    options: GraphSearchOptions,
  ): Promise<TripleMatch[]>;
  /** Drop every triple linked to `memoryId` within the given filter scope. */
  abstract deleteByMemoryId(
    memoryId: string,
    filters: Record<string, unknown>,
  ): Promise<void>;
  abstract reset(): Promise<void>;

  /** Release backing resources (DB handles, sockets). No-op by default. */
  close(): void {
    /* no-op */
  }
}
