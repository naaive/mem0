import type { LLMConfig, Message } from "../types";

export interface LLMResponseOptions {
  /** Hint to the provider that response should be JSON. */
  responseFormat?: "json_object" | "text";
}

export abstract class LLM {
  protected readonly config: LLMConfig;

  constructor(config: LLMConfig = {}) {
    this.config = config;
  }

  abstract generateResponse(
    messages: Message[],
    options?: LLMResponseOptions,
  ): Promise<string>;
}
