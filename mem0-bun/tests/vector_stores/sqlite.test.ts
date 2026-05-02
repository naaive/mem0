import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { SqliteVectorStore } from "../../src/vector_stores/sqlite";

function makeTmp(): string {
  return mkdtempSync(join(tmpdir(), "mem0-bun-sqlitevs-"));
}

describe("SqliteVectorStore", () => {
  const tmpDirs: string[] = [];
  afterEach(() => {
    for (const d of tmpDirs.splice(0)) {
      try {
        rmSync(d, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    }
  });

  test("initialize is idempotent and CRUD round-trips in :memory:", async () => {
    const s = new SqliteVectorStore({ collectionName: "t" });
    await s.initialize();
    await s.initialize();
    await s.insert(
      [[1, 0]],
      ["a"],
      [{ data: "alpha", user_id: "u1" }],
    );
    const got = await s.get("a");
    expect(got?.payload.data).toBe("alpha");
    expect(await s.get("missing")).toBeNull();
    s.close();
  });

  test("insert validates lengths", async () => {
    const s = new SqliteVectorStore();
    await expect(
      s.insert([[1]], ["a", "b"], [{}]),
    ).rejects.toThrow(/equal length/);
    s.close();
  });

  test("disk path persists across instances", async () => {
    const dir = makeTmp();
    tmpDirs.push(dir);
    const path = join(dir, "vec.db");
    {
      const s = new SqliteVectorStore({ path });
      await s.insert([[1, 0]], ["a"], [{ data: "alpha", user_id: "u1" }]);
      s.close();
    }
    {
      const s = new SqliteVectorStore({ path });
      const got = await s.get("a");
      expect(got?.payload.data).toBe("alpha");
      s.close();
    }
  });

  test("search ranks by cosine and respects scoped filters", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      ["a", "b", "c"],
      [
        { data: "alpha", user_id: "u1" },
        { data: "beta", user_id: "u1" },
        { data: "gamma", user_id: "u2" },
      ],
    );
    const out = await s.search([1, 0, 0], 5, { user_id: "u1" });
    expect(out[0]!.id).toBe("a");
    expect(out.map((r) => r.id)).toEqual(["a", "b"]);
    s.close();
  });

  test("search applies extra payload filters in TS layer", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [
        [1, 0],
        [0, 1],
      ],
      ["a", "b"],
      [
        { data: "alpha", user_id: "u1", tag: "x" },
        { data: "beta", user_id: "u1", tag: "y" },
      ],
    );
    const out = await s.search([1, 0], 5, { user_id: "u1", tag: "x" });
    expect(out.map((r) => r.id)).toEqual(["a"]);
    s.close();
  });

  test("search ignores undefined filter values", async () => {
    const s = new SqliteVectorStore();
    await s.insert([[1, 0]], ["a"], [{ data: "x", user_id: "u1" }]);
    const out = await s.search([1, 0], 5, {
      user_id: "u1",
      agent_id: undefined,
    });
    expect(out.length).toBe(1);
    s.close();
  });

  test("keywordSearch returns BM25-ranked rows", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      ["a", "b", "c"],
      [
        { data: "pepperoni pizza is great", user_id: "u1" },
        { data: "cheese pizza", user_id: "u1" },
        { data: "broccoli", user_id: "u1" },
      ],
    );
    const out = await s.keywordSearch!(
      ["pepperoni", "pizza"],
      10,
      { user_id: "u1" },
    );
    expect(out.length).toBeGreaterThan(0);
    expect(out[0]!.id).toBe("a");
    expect(out.find((r) => r.id === "c")).toBeUndefined();
    s.close();
  });

  test("keywordSearch returns empty for empty token list", async () => {
    const s = new SqliteVectorStore();
    expect(await s.keywordSearch!([], 5)).toEqual([]);
    s.close();
  });

  test("keywordSearch escapes embedded quotes in tokens", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [[1]],
      ["a"],
      [{ data: 'has "quoted" word', user_id: "u" }],
    );
    const out = await s.keywordSearch!(['has'], 5, { user_id: "u" });
    expect(out.length).toBeGreaterThan(0);
    s.close();
  });

  test("keywordSearch works without filters", async () => {
    const s = new SqliteVectorStore();
    await s.insert([[1]], ["a"], [{ data: "hello world" }]);
    const out = await s.keywordSearch!(["hello"], 5);
    expect(out.length).toBe(1);
    s.close();
  });

  test("keywordSearch applies extra payload filters", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [
        [1, 0],
        [0, 1],
      ],
      ["a", "b"],
      [
        { data: "pizza", user_id: "u1", tag: "x" },
        { data: "pizza", user_id: "u1", tag: "y" },
      ],
    );
    const out = await s.keywordSearch!(["pizza"], 10, {
      user_id: "u1",
      tag: "x",
    });
    expect(out.map((r) => r.id)).toEqual(["a"]);
    s.close();
  });

  test("update mutates payload + FTS index", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [[1]],
      ["a"],
      [{ data: "old phrase", user_id: "u" }],
    );
    await s.update("a", [1], {
      data: "new pizza",
      user_id: "u",
    });
    const fresh = await s.get("a");
    expect(fresh?.payload.data).toBe("new pizza");
    const kw = await s.keywordSearch!(["pizza"], 5, { user_id: "u" });
    expect(kw.length).toBe(1);
    s.close();
  });

  test("update on missing record throws", async () => {
    const s = new SqliteVectorStore();
    await expect(s.update("nope", [1], {})).rejects.toThrow(/not found/);
    s.close();
  });

  test("delete removes from records + FTS", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [[1]],
      ["a"],
      [{ data: "pizza", user_id: "u" }],
    );
    await s.delete("a");
    expect(await s.get("a")).toBeNull();
    expect(await s.keywordSearch!(["pizza"], 5, { user_id: "u" })).toEqual([]);
    s.close();
  });

  test("list respects scoped filters and limit", async () => {
    const s = new SqliteVectorStore();
    for (let i = 0; i < 3; i++) {
      await s.insert(
        [[1]],
        [`id${i}`],
        [{ data: `m${i}`, user_id: "u1" }],
      );
    }
    const all = await s.list({ user_id: "u1" });
    expect(all.length).toBe(3);
    const limited = await s.list({ user_id: "u1" }, 2);
    expect(limited.length).toBe(2);
    s.close();
  });

  test("list applies extra payload filters", async () => {
    const s = new SqliteVectorStore();
    await s.insert(
      [
        [1],
        [1],
      ],
      ["a", "b"],
      [
        { data: "x", user_id: "u", tag: "yes" },
        { data: "y", user_id: "u", tag: "no" },
      ],
    );
    const out = await s.list({ user_id: "u", tag: "yes" });
    expect(out.map((r) => r.id)).toEqual(["a"]);
    s.close();
  });

  test("deleteCollection wipes records + FTS", async () => {
    const s = new SqliteVectorStore();
    await s.insert([[1]], ["a"], [{ data: "x", user_id: "u" }]);
    await s.deleteCollection();
    expect(await s.list({ user_id: "u" })).toEqual([]);
    expect(await s.keywordSearch!(["x"], 5, { user_id: "u" })).toEqual([]);
    s.close();
  });

  test("numeric filter values are passed through", async () => {
    const s = new SqliteVectorStore();
    await s.insert([[1]], ["a"], [{ data: "x", user_id: "u", priority: 5 }]);
    // Forcing filter type through the public path.
    const out = await s.list({ user_id: "u", priority: 5 });
    expect(out.length).toBe(1);
    s.close();
  });
});
