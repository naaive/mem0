import { LLM, type LLMResponseOptions } from "./base";
import type { LLMConfig, Message } from "../types";

/**
 * Test-only LLM that returns canned responses, optionally selected by a
 * matcher function. Mock providers like this are exposed publicly so users
 * can write deterministic tests against the Memory class.
 */
export type MockLLMResponder = (
  messages: Message[],
  options: LLMResponseOptions,
) => string | Promise<string>;

export class MockLLM extends LLM {
  private readonly responder: MockLLMResponder;

  constructor(
    config: LLMConfig & { responder?: MockLLMResponder } = {},
  ) {
    super(config);
    this.responder = config.responder ?? (() => "");
  }

  async generateResponse(
    messages: Message[],
    options: LLMResponseOptions = {},
  ): Promise<string> {
    return await this.responder(messages, options);
  }
}
