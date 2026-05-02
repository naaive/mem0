import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { AnthropicLLM } from "../../src/llms/anthropic";
import { mockFetchSequence } from "../_helpers/fetch_mock";

describe("AnthropicLLM", () => {
  const orig = process.env.ANTHROPIC_API_KEY;

  beforeEach(() => {
    delete process.env.ANTHROPIC_API_KEY;
  });
  afterEach(() => {
    if (orig !== undefined) process.env.ANTHROPIC_API_KEY = orig;
    else delete process.env.ANTHROPIC_API_KEY;
  });

  test("requires apiKey", () => {
    expect(() => new AnthropicLLM()).toThrow(/apiKey/);
  });

  test("falls back to env", () => {
    process.env.ANTHROPIC_API_KEY = "envk";
    expect(new AnthropicLLM()).toBeDefined();
  });

  test("splits system messages and posts to /messages", async () => {
    const m = mockFetchSequence([
      {
        body: { content: [{ type: "text", text: "hi" }] },
      },
    ]);
    try {
      const llm = new AnthropicLLM({
        apiKey: "k",
        model: "claude-x",
        baseURL: "https://api.example.com/v1",
        topP: 0.8,
        temperature: 0.2,
        maxTokens: 50,
      });
      const out = await llm.generateResponse(
        [
          { role: "system", content: "be terse" },
          { role: "user", content: "hello" },
          { role: "assistant", content: "hi" },
          { role: "tool", content: "extra" },
        ],
        { responseFormat: "json_object" },
      );
      expect(out).toBe("hi");
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.system).toContain("be terse");
      expect(body.system).toContain("strictly valid JSON");
      expect(body.messages).toEqual([
        { role: "user", content: "hello" },
        { role: "assistant", content: "hi" },
        { role: "user", content: "extra" }, // non-assistant non-system maps to user
      ]);
      expect(body.model).toBe("claude-x");
      expect(body.top_p).toBe(0.8);
      expect(body.max_tokens).toBe(50);
      expect(m.calls[0]!.url).toBe("https://api.example.com/v1/messages");
      expect(m.calls[0]!.headers["x-api-key"]).toBe("k");
      expect(m.calls[0]!.headers["anthropic-version"]).toBe("2023-06-01");
    } finally {
      m.dispose();
    }
  });

  test("uses defaults when no extras", async () => {
    const m = mockFetchSequence([
      { body: { content: [{ type: "text", text: "ok" }] } },
    ]);
    try {
      const llm = new AnthropicLLM({ apiKey: "k" });
      await llm.generateResponse([{ role: "user", content: "hi" }]);
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.top_p).toBeUndefined();
      expect(body.max_tokens).toBe(2048);
      expect(body.model).toBe("claude-sonnet-4-6");
      expect(body.system).toBe("");
    } finally {
      m.dispose();
    }
  });

  test("throws on non-ok response", async () => {
    const m = mockFetchSequence([{ status: 401, bodyText: "bad" }]);
    try {
      const llm = new AnthropicLLM({ apiKey: "k" });
      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow(/Anthropic LLM error 401/);
    } finally {
      m.dispose();
    }
  });

  test("throws when no text content", async () => {
    const m = mockFetchSequence([
      { body: { content: [{ type: "image", text: "" }] } },
    ]);
    try {
      const llm = new AnthropicLLM({ apiKey: "k" });
      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow(/no text content/);
    } finally {
      m.dispose();
    }
  });

  test("throws when content array missing", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const llm = new AnthropicLLM({ apiKey: "k" });
      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow(/no text content/);
    } finally {
      m.dispose();
    }
  });
});
