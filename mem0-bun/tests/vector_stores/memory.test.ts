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
});
