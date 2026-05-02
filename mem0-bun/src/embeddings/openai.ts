import { Embedder } from "./base";
import type { EmbedderConfig } from "../types";

export class OpenAIEmbedder extends Embedder {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseURL: string;

  constructor(config: EmbedderConfig = {}) {
    super(config);
    const apiKey = config.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "OpenAI Embedder requires an apiKey (or OPENAI_API_KEY env)",
      );
    }
    this.apiKey = apiKey;
    this.model = config.model ?? "text-embedding-3-small";
    this.baseURL = config.baseURL ?? "https://api.openai.com/v1";
  }

  async embed(text: string): Promise<number[]> {
    const result = await this.embedBatch([text]);
    return result[0]!;
  }

  override async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const res = await fetch(`${this.baseURL}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({ model: this.model, input: texts }),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`OpenAI Embedder error ${res.status}: ${errText}`);
    }
    const data = (await res.json()) as {
      data?: Array<{ embedding?: number[] }>;
    };
    const embeddings = data.data?.map((d) => d.embedding ?? []);
    if (!embeddings || embeddings.length !== texts.length) {
      throw new Error("OpenAI Embedder returned invalid response");
    }
    return embeddings;
  }
}
