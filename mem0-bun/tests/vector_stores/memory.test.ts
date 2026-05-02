import { describe, expect, test } from "bun:test";
import { InMemoryVectorStore } from "../../src/vector_stores/memory";

describe("InMemoryVectorStore", () => {
  const baseConfig = { collectionName: "t" };

  test("default collection name is used when none provided", async () => {
    const s = new InMemoryVectorStore();
    await s.initialize();
    await s.insert([[1]], ["x"], [{ data: "a" }]);
    const got = await s.get("x");
    expect(got?.payload.data).toBe("a");
  });

  test("insert validates lengths", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await expect(
      s.insert([[1]], ["a", "b"], [{}]),
    ).rejects.toThrow(/equal length/);
  });

  test("CRUD round trip", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert(
      [[1, 0], [0, 1]],
      ["a", "b"],
      [
        { data: "alpha", user_id: "u1" },
        { data: "beta", user_id: "u1" },
      ],
    );
    expect((await s.get("a"))?.payload.data).toBe("alpha");
    expect(await s.get("missing")).toBeNull();

    const search = await s.search([1, 0], 2, { user_id: "u1" });
    expect(search[0]!.id).toBe("a");
    expect(search[0]!.score).toBeGreaterThan(0.99);
    expect(search.length).toBe(2);

    const list = await s.list({ user_id: "u1" });
    expect(list.length).toBe(2);

    await s.update("a", [0, 1], { data: "alpha2", user_id: "u1" });
    expect((await s.get("a"))?.payload.data).toBe("alpha2");

    await s.delete("a");
    expect(await s.get("a")).toBeNull();
  });

  test("search and list ignore mismatched filters", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert([[1, 0]], ["a"], [{ data: "x", user_id: "u1" }]);
    expect(await s.search([1, 0], 5, { user_id: "u2" })).toHaveLength(0);
    expect(await s.list({ user_id: "u2" })).toHaveLength(0);
    expect(await s.list()).toHaveLength(1);
  });

  test("undefined filter values are skipped", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert([[1, 0]], ["a"], [{ data: "x", user_id: "u1" }]);
    const out = await s.search([1, 0], 5, {
      user_id: "u1",
      agent_id: undefined,
    });
    expect(out).toHaveLength(1);
  });

  test("list respects limit", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert(
      [[1], [1], [1]],
      ["a", "b", "c"],
      [{}, {}, {}],
    );
    expect((await s.list(undefined, 2)).length).toBe(2);
  });

  test("update on missing record throws", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await expect(s.update("nope", [1], {})).rejects.toThrow(/not found/);
  });

  test("deleteCollection clears data", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert([[1]], ["a"], [{}]);
    await s.deleteCollection();
    expect(await s.list()).toHaveLength(0);
  });

  test("delete on missing id is a no-op", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.delete("ghost");
  });

  test("keywordSearch ranks documents by BM25 over textLemmatized", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert(
      [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      ["a", "b", "c"],
      [
        { data: "pizza", textLemmatized: ["pizza", "pepperoni"], user_id: "u1" },
        { data: "pizza", textLemmatized: ["pizza"], user_id: "u1" },
        { data: "broccoli", textLemmatized: ["broccoli"], user_id: "u1" },
      ],
    );
    const out = await s.keywordSearch!(
      ["pizza", "pepperoni"],
      10,
      { user_id: "u1" },
    );
    expect(out.length).toBeGreaterThan(0);
    expect(out[0]!.id).toBe("a");
    expect(out.find((r) => r.id === "c")).toBeUndefined();
  });

  test("keywordSearch handles records without textLemmatized payloads", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert([[1]], ["a"], [{ data: "no tokens" }]);
    expect(await s.keywordSearch!(["pizza"], 5)).toEqual([]);
  });

  test("update invalidates BM25 stats so re-tokenized docs surface", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert([[1]], ["a"], [
      { data: "first", textLemmatized: ["first"], user_id: "u" },
    ]);
    await s.update("a", [1], {
      data: "second",
      textLemmatized: ["pizza"],
      user_id: "u",
    });
    const out = await s.keywordSearch!(["pizza"], 5, { user_id: "u" });
    expect(out.length).toBe(1);
  });

  test("delete invalidates BM25 stats", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.insert(
      [[1], [1]],
      ["a", "b"],
      [
        { data: "x", textLemmatized: ["pizza"], user_id: "u" },
        { data: "y", textLemmatized: ["broccoli"], user_id: "u" },
      ],
    );
    await s.delete("b");
    expect(await s.keywordSearch!(["broccoli"], 5, { user_id: "u" })).toEqual(
      [],
    );
  });

  test("delete of missing id does not invalidate stats but is harmless", async () => {
    const s = new InMemoryVectorStore(baseConfig);
    await s.initialize();
    await s.delete("ghost");
    expect(await s.list()).toEqual([]);
  });
});
