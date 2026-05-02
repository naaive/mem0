import { describe, expect, test } from "bun:test";
import { OllamaEmbedder } from "../../src/embeddings/ollama";
import { mockFetchSequence } from "../_helpers/fetch_mock";

describe("OllamaEmbedder", () => {
  test("calls /api/embeddings with default model", async () => {
    const m = mockFetchSequence([
      { body: { embedding: [0.1, 0.2] } },
    ]);
    try {
      const e = new OllamaEmbedder({ baseURL: "http://localhost:9999" });
      const out = await e.embed("hi");
      expect(out).toEqual([0.1, 0.2]);
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.model).toBe("nomic-embed-text");
      expect(body.prompt).toBe("hi");
      expect(m.calls[0]!.url).toBe(
        "http://localhost:9999/api/embeddings",
      );
    } finally {
      m.dispose();
    }
  });

  test("uses configured model and default URL", async () => {
    const m = mockFetchSequence([
      { body: { embedding: [1] } },
    ]);
    try {
      const e = new OllamaEmbedder({ model: "x" });
      await e.embed("hi");
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.model).toBe("x");
      expect(m.calls[0]!.url).toBe(
        "http://localhost:11434/api/embeddings",
      );
    } finally {
      m.dispose();
    }
  });

  test("throws on non-ok response", async () => {
    const m = mockFetchSequence([{ status: 503, bodyText: "down" }]);
    try {
      const e = new OllamaEmbedder();
      await expect(e.embed("hi")).rejects.toThrow(/Embedder error 503/);
    } finally {
      m.dispose();
    }
  });

  test("throws on missing embedding", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const e = new OllamaEmbedder();
      await expect(e.embed("hi")).rejects.toThrow(/no embedding/);
    } finally {
      m.dispose();
    }
  });

  test("throws on empty embedding", async () => {
    const m = mockFetchSequence([{ body: { embedding: [] } }]);
    try {
      const e = new OllamaEmbedder();
      await expect(e.embed("hi")).rejects.toThrow(/no embedding/);
    } finally {
      m.dispose();
    }
  });
});
