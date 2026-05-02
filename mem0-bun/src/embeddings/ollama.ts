import { Embedder } from "./base";
import type { EmbedderConfig } from "../types";

export class OllamaEmbedder extends Embedder {
  private readonly model: string;
  private readonly baseURL: string;

  constructor(config: EmbedderConfig = {}) {
    super(config);
    this.model = config.model ?? "nomic-embed-text";
    this.baseURL = config.baseURL ?? "http://localhost:11434";
  }

  async embed(text: string): Promise<number[]> {
    const res = await fetch(`${this.baseURL}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, prompt: text }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Ollama Embedder error ${res.status}: ${errText}`);
    }
    const data = (await res.json()) as { embedding?: number[] };
    if (!Array.isArray(data.embedding) || data.embedding.length === 0) {
      throw new Error("Ollama Embedder returned no embedding");
    }
    return data.embedding;
  }
}
