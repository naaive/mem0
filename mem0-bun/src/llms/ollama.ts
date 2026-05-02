import { LLM, type LLMResponseOptions } from "./base";
import type { LLMConfig, Message } from "../types";

export class OllamaLLM extends LLM {
  private readonly model: string;
  private readonly baseURL: string;

  constructor(config: LLMConfig = {}) {
    super(config);
    this.model = config.model ?? "llama3";
    this.baseURL = config.baseURL ?? "http://localhost:11434";
  }

  async generateResponse(
    messages: Message[],
    options: LLMResponseOptions = {},
  ): Promise<string> {
    const body: Record<string, unknown> = {
      model: this.model,
      messages,
      stream: false,
      options: {
        temperature: this.config.temperature ?? 0.1,
      },
    };
    if (options.responseFormat === "json_object") {
      body.format = "json";
    }

    const res = await fetch(`${this.baseURL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Ollama LLM error ${res.status}: ${errText}`);
    }
    const data = (await res.json()) as { message?: { content?: string } };
    const content = data.message?.content;
    if (typeof content !== "string") {
      throw new Error("Ollama LLM returned no content");
    }
    return content;
  }
}
