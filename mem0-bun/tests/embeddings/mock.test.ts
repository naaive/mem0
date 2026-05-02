import { describe, expect, test } from "bun:test";
import { MockEmbedder } from "../../src/embeddings/mock";

describe("MockEmbedder", () => {
  test("uses default dimension", async () => {
    const e = new MockEmbedder();
    const v = await e.embed("hello");
    expect(v.length).toBe(16);
  });

  test("respects configured dimension", async () => {
    const e = new MockEmbedder({ embeddingDims: 8 });
    const v = await e.embed("test");
    expect(v.length).toBe(8);
  });

  test("empty text yields zero vector", async () => {
    const e = new MockEmbedder({ embeddingDims: 4 });
    expect(await e.embed("")).toEqual([0, 0, 0, 0]);
  });

  test("same text yields same vector", async () => {
    const e = new MockEmbedder({ embeddingDims: 4 });
    expect(await e.embed("abc")).toEqual(await e.embed("abc"));
  });

  test("embedBatch returns vectors per item via base", async () => {
    const e = new MockEmbedder({ embeddingDims: 4 });
    const out = await e.embedBatch(["a", "b"]);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBe(4);
  });
});
