import { describe, expect, test } from "bun:test";
import { OllamaLLM } from "../../src/llms/ollama";
import { mockFetchSequence } from "../_helpers/fetch_mock";

describe("OllamaLLM", () => {
  test("posts to /api/chat with default model and json format", async () => {
    const m = mockFetchSequence([
      { body: { message: { content: "ok" } } },
    ]);
    try {
      const llm = new OllamaLLM({ baseURL: "http://localhost:1234" });
      const out = await llm.generateResponse(
        [{ role: "user", content: "hi" }],
        { responseFormat: "json_object" },
      );
      expect(out).toBe("ok");
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.format).toBe("json");
      expect(body.model).toBe("llama3");
      expect(body.options.temperature).toBe(0.1);
      expect(m.calls[0]!.url).toBe("http://localhost:1234/api/chat");
    } finally {
      m.dispose();
    }
  });

  test("uses configured model", async () => {
    const m = mockFetchSequence([
      { body: { message: { content: "yo" } } },
    ]);
    try {
      const llm = new OllamaLLM({ model: "qwen2", temperature: 0.7 });
      await llm.generateResponse([{ role: "user", content: "hi" }]);
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.model).toBe("qwen2");
      expect(body.options.temperature).toBe(0.7);
      expect(body.format).toBeUndefined();
    } finally {
      m.dispose();
    }
  });

  test("throws on non-ok response", async () => {
    const m = mockFetchSequence([{ status: 502, bodyText: "fail" }]);
    try {
      const llm = new OllamaLLM();
      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow(/Ollama LLM error 502/);
    } finally {
      m.dispose();
    }
  });

  test("throws when no content", async () => {
    const m = mockFetchSequence([{ body: { message: {} } }]);
    try {
      const llm = new OllamaLLM();
      await expect(
        llm.generateResponse([{ role: "user", content: "hi" }]),
      ).rejects.toThrow(/no content/);
    } finally {
      m.dispose();
    }
  });
});
