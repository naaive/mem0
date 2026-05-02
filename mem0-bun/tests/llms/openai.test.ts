import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { OpenAILLM } from "../../src/llms/openai";
import { mockFetchSequence } from "../_helpers/fetch_mock";

describe("OpenAILLM", () => {
  const originalEnv = process.env.OPENAI_API_KEY;

  beforeEach(() => {
    delete process.env.OPENAI_API_KEY;
  });
  afterEach(() => {
    if (originalEnv !== undefined) process.env.OPENAI_API_KEY = originalEnv;
    else delete process.env.OPENAI_API_KEY;
  });

  test("requires apiKey", () => {
    expect(() => new OpenAILLM()).toThrow(/apiKey/);
  });

  test("falls back to OPENAI_API_KEY env", () => {
    process.env.OPENAI_API_KEY = "envk";
    const llm = new OpenAILLM();
    expect(llm).toBeDefined();
  });

  test("calls /chat/completions with default model and json_object format", async () => {
    const m = mockFetchSequence([
      {
        body: { choices: [{ message: { content: "ok" } }] },
      },
    ]);
    try {
      const llm = new OpenAILLM({
        apiKey: "k",
        model: "gpt-4o",
        baseURL: "https://api.example.com/v1",
        maxTokens: 100,
        topP: 0.9,
        temperature: 0.5,
      });
      const out = await llm.generateResponse(
        [{ role: "user", content: "hello" }],
        { responseFormat: "json_object" },
      );
      expect(out).toBe("ok");
      expect(m.calls).toHaveLength(1);
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.model).toBe("gpt-4o");
      expect(body.response_format).toEqual({ type: "json_object" });
      expect(body.max_tokens).toBe(100);
      expect(body.top_p).toBe(0.9);
      expect(m.calls[0]!.url).toBe("https://api.example.com/v1/chat/completions");
      expect(m.calls[0]!.headers.authorization).toBe("Bearer k");
    } finally {
      m.dispose();
    }
  });

  test("uses defaults when extras are omitted", async () => {
    const m = mockFetchSequence([
      { body: { choices: [{ message: { content: "ok" } }] } },
    ]);
    try {
      const llm = new OpenAILLM({ apiKey: "k" });
      await llm.generateResponse([{ role: "user", content: "hi" }]);
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.max_tokens).toBeUndefined();
      expect(body.top_p).toBeUndefined();
      expect(body.response_format).toBeUndefined();
      expect(body.model).toBe("gpt-4o-mini");
      expect(body.temperature).toBe(0.1);
    } finally {
      m.dispose();
    }
  });

  test("throws on non-ok response", async () => {
    const m = mockFetchSequence([{ status: 500, bodyText: "oops" }]);
    try {
      const llm = new OpenAILLM({ apiKey: "k" });
      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow(/OpenAI LLM error 500/);
    } finally {
      m.dispose();
    }
  });

  test("throws when no content returned", async () => {
    const m = mockFetchSequence([{ body: { choices: [{}] } }]);
    try {
      const llm = new OpenAILLM({ apiKey: "k" });
      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow(/no content/);
    } finally {
      m.dispose();
    }
  });
});
