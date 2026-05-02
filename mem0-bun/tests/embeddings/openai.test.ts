import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { OpenAIEmbedder } from "../../src/embeddings/openai";
import { mockFetchSequence } from "../_helpers/fetch_mock";

describe("OpenAIEmbedder", () => {
  const orig = process.env.OPENAI_API_KEY;
  beforeEach(() => {
    delete process.env.OPENAI_API_KEY;
  });
  afterEach(() => {
    if (orig !== undefined) process.env.OPENAI_API_KEY = orig;
    else delete process.env.OPENAI_API_KEY;
  });

  test("requires apiKey", () => {
    expect(() => new OpenAIEmbedder()).toThrow(/apiKey/);
  });

  test("falls back to env", () => {
    process.env.OPENAI_API_KEY = "k";
    expect(new OpenAIEmbedder()).toBeDefined();
  });

  test("embed returns first vector", async () => {
    const m = mockFetchSequence([
      { body: { data: [{ embedding: [1, 2, 3] }] } },
    ]);
    try {
      const e = new OpenAIEmbedder({
        apiKey: "k",
        model: "m",
        baseURL: "https://x/v1",
      });
      const out = await e.embed("hello");
      expect(out).toEqual([1, 2, 3]);
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.input).toEqual(["hello"]);
      expect(body.model).toBe("m");
      expect(m.calls[0]!.url).toBe("https://x/v1/embeddings");
      expect(m.calls[0]!.headers.authorization).toBe("Bearer k");
    } finally {
      m.dispose();
    }
  });

  test("embedBatch returns vectors in order", async () => {
    const m = mockFetchSequence([
      {
        body: {
          data: [{ embedding: [1] }, { embedding: [2] }],
        },
      },
    ]);
    try {
      const e = new OpenAIEmbedder({ apiKey: "k" });
      const out = await e.embedBatch(["a", "b"]);
      expect(out).toEqual([[1], [2]]);
    } finally {
      m.dispose();
    }
  });

  test("embedBatch returns empty for empty input without HTTP call", async () => {
    const m = mockFetchSequence([]);
    try {
      const e = new OpenAIEmbedder({ apiKey: "k" });
      const out = await e.embedBatch([]);
      expect(out).toEqual([]);
      expect(m.calls).toHaveLength(0);
    } finally {
      m.dispose();
    }
  });

  test("throws on non-ok response", async () => {
    const m = mockFetchSequence([{ status: 500, bodyText: "fail" }]);
    try {
      const e = new OpenAIEmbedder({ apiKey: "k" });
      await expect(e.embed("hi")).rejects.toThrow(/Embedder error 500/);
    } finally {
      m.dispose();
    }
  });

  test("throws on invalid response shape", async () => {
    const m = mockFetchSequence([{ body: { data: [] } }]);
    try {
      const e = new OpenAIEmbedder({ apiKey: "k" });
      await expect(e.embed("hi")).rejects.toThrow(/invalid response/);
    } finally {
      m.dispose();
    }
  });

  test("uses default model when not specified", async () => {
    const m = mockFetchSequence([
      { body: { data: [{ embedding: [1, 2] }] } },
    ]);
    try {
      const e = new OpenAIEmbedder({ apiKey: "k" });
      await e.embed("hi");
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.model).toBe("text-embedding-3-small");
      expect(m.calls[0]!.url).toBe(
        "https://api.openai.com/v1/embeddings",
      );
    } finally {
      m.dispose();
    }
  });
});
