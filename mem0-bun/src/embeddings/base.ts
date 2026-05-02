import type { EmbedderConfig } from "../types";

export abstract class Embedder {
  protected readonly config: EmbedderConfig;

  constructor(config: EmbedderConfig = {}) {
    this.config = config;
  }

  abstract embed(text: string): Promise<number[]>;

  async embedBatch(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (const t of texts) out.push(await this.embed(t));
    return out;
  }
}
