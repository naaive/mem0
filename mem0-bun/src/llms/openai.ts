import { LLM, type LLMResponseOptions } from "./base";
import type { LLMConfig, Message } from "../types";

export class OpenAILLM extends LLM {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly baseURL: string;

  constructor(config: LLMConfig = {}) {
    super(config);
    const apiKey = config.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OpenAI LLM requires an apiKey (or OPENAI_API_KEY env)");
    }
    this.apiKey = apiKey;
    this.model = config.model ?? "gpt-4o-mini";
    this.baseURL = config.baseURL ?? "https://api.openai.com/v1";
  }

  async generateResponse(
    messages: Message[],
    options: LLMResponseOptions = {},
  ): Promise<string> {
    const body: Record<string, unknown> = {
      model: this.model,
      messages,
      temperature: this.config.temperature ?? 0.1,
    };
    if (this.config.maxTokens !== undefined) {
      body.max_tokens = this.config.maxTokens;
    }
    if (this.config.topP !== undefined) {
      body.top_p = this.config.topP;
    }
    if (options.responseFormat === "json_object") {
      body.response_format = { type: "json_object" };
    }

    const res = await fetch(`${this.baseURL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`OpenAI LLM error ${res.status}: ${errText}`);
    }
    const data = (await res.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    const content = data.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      throw new Error("OpenAI LLM returned no content");
    }
    return content;
  }
}
