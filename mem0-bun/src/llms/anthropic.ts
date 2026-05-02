import { LLM, type LLMResponseOptions } from "./base";
import type { LLMConfig, Message } from "../types";

export class AnthropicLLM extends LLM {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseURL: string;

  constructor(config: LLMConfig = {}) {
    super(config);
    const apiKey = config.apiKey ?? process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error(
        "Anthropic LLM requires an apiKey (or ANTHROPIC_API_KEY env)",
      );
    }
    this.apiKey = apiKey;
    this.model = config.model ?? "claude-sonnet-4-6";
    this.baseURL = config.baseURL ?? "https://api.anthropic.com/v1";
  }

  async generateResponse(
    messages: Message[],
    options: LLMResponseOptions = {},
  ): Promise<string> {
    const system = messages
      .filter((m) => m.role === "system")
      .map((m) => m.content)
      .join("\n");
    const conversation = messages.filter((m) => m.role !== "system");

    const effectiveSystem =
      options.responseFormat === "json_object"
        ? `${system}\n\nRespond with strictly valid JSON only.`
        : system;

    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: this.config.maxTokens ?? 2048,
      temperature: this.config.temperature ?? 0.1,
      system: effectiveSystem,
      messages: conversation.map((m) => ({
        role: m.role === "assistant" ? "assistant" : "user",
        content: m.content,
      })),
    };
    if (this.config.topP !== undefined) body.top_p = this.config.topP;

    const res = await fetch(`${this.baseURL}/messages`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Anthropic LLM error ${res.status}: ${errText}`);
    }
    const data = (await res.json()) as {
      content?: Array<{ type: string; text?: string }>;
    };
    const text = data.content
      ?.filter((b) => b.type === "text")
      .map((b) => b.text ?? "")
      .join("");
    if (!text) {
      throw new Error("Anthropic LLM returned no text content");
    }
    return text;
  }
}
