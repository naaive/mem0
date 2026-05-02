import { Embedder } from "./base";
import type { EmbedderConfig } from "../types";

/**
 * Deterministic embedder for tests. Maps each unique character to a slot
 * in a fixed-dimension vector and counts occurrences. Cosine similarity
 * is well-defined and reproducible across runs.
 */
export class MockEmbedder extends Embedder {
  private readonly dimension: number;

  constructor(config: EmbedderConfig = {}) {
    super(config);
    this.dimension = config.embeddingDims ?? 16;
  }

  async embed(text: string): Promise<number[]> {
    const vec = new Array<number>(this.dimension).fill(0);
    if (!text) return vec;
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      vec[code % this.dimension]! += 1;
    }
    return vec;
  }
}
