import { describe, expect, test } from "bun:test";
import { QdrantVectorStore } from "../../src/vector_stores/qdrant";
import { mockFetchHandler, mockFetchSequence } from "../_helpers/fetch_mock";

describe("QdrantVectorStore", () => {
  test("requires dimension", () => {
    expect(() => new QdrantVectorStore({})).toThrow(/dimension/);
  });

  test("strips trailing slash from baseURL", () => {
    const s = new QdrantVectorStore({
      url: "http://localhost:6333/",
      dimension: 4,
    });
    expect(s).toBeDefined();
  });

  test("initialize creates collection when missing", async () => {
    const m = mockFetchSequence([
      { body: { result: { exists: false } } },
      { body: { result: true } },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        apiKey: "k",
        collectionName: "col",
        dimension: 4,
      });
      await s.initialize();
      expect(m.calls).toHaveLength(2);
      expect(m.calls[0]!.method).toBe("GET");
      expect(m.calls[0]!.url).toBe("http://q.test/collections/col/exists");
      expect(m.calls[0]!.headers["api-key"]).toBe("k");
      expect(m.calls[1]!.method).toBe("PUT");
      const createBody = JSON.parse(m.calls[1]!.body!);
      expect(createBody.vectors).toEqual({ size: 4, distance: "Cosine" });
    } finally {
      m.dispose();
    }
  });

  test("initialize is no-op when collection already exists", async () => {
    const m = mockFetchSequence([
      { body: { result: { exists: true } } },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 4,
      });
      await s.initialize();
      expect(m.calls).toHaveLength(1);
    } finally {
      m.dispose();
    }
  });

  test("insert sends points and includes filter on search", async () => {
    const m = mockFetchHandler(async (url) => {
      if (url.endsWith("/points?wait=true")) {
        return { body: { result: true } };
      }
      if (url.endsWith("/points/search")) {
        return {
          body: {
            result: [
              { id: 7, payload: { data: "x" }, score: 0.9 },
            ],
          },
        };
      }
      return { body: {} };
    });
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      await s.insert([[1, 0]], ["a"], [{ data: "x" }]);
      const out = await s.search([1, 0], 1, { user_id: "u1" });
      expect(out).toEqual([
        { id: "7", payload: { data: "x" }, score: 0.9 },
      ]);
      const searchCall = m.calls.find((c) => c.url.endsWith("/points/search"))!;
      const body = JSON.parse(searchCall.body!);
      expect(body.filter.must).toEqual([
        { key: "user_id", match: { value: "u1" } },
      ]);
    } finally {
      m.dispose();
    }
  });

  test("search omits filter when no defined entries", async () => {
    const m = mockFetchSequence([
      { body: { result: [] } },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      await s.search([1, 0], 1, { user_id: undefined });
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.filter).toBeUndefined();
    } finally {
      m.dispose();
    }
  });

  test("search returns empty result list when result is missing", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      const out = await s.search([1, 0], 5);
      expect(out).toEqual([]);
    } finally {
      m.dispose();
    }
  });

  test("get returns record when present", async () => {
    const m = mockFetchSequence([
      { body: { result: { id: "abc", payload: { data: "x" } } } },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      const out = await s.get("abc");
      expect(out).toEqual({ id: "abc", payload: { data: "x" } });
    } finally {
      m.dispose();
    }
  });

  test("get returns null when missing", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      const out = await s.get("abc");
      expect(out).toBeNull();
    } finally {
      m.dispose();
    }
  });

  test("get swallows errors and returns null", async () => {
    const m = mockFetchSequence([{ status: 500, bodyText: "boom" }]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      expect(await s.get("abc")).toBeNull();
    } finally {
      m.dispose();
    }
  });

  test("get returns null when payload missing", async () => {
    const m = mockFetchSequence([
      { body: { result: { id: "abc" } } },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      const out = await s.get("abc");
      expect(out).toEqual({ id: "abc", payload: {} });
    } finally {
      m.dispose();
    }
  });

  test("update reuses insert", async () => {
    const m = mockFetchSequence([
      { body: { result: true } },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      await s.update("a", [1, 0], { data: "x" });
      expect(m.calls[0]!.method).toBe("PUT");
    } finally {
      m.dispose();
    }
  });

  test("delete posts ids", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      await s.delete("a");
      const body = JSON.parse(m.calls[0]!.body!);
      expect(body.points).toEqual(["a"]);
    } finally {
      m.dispose();
    }
  });

  test("list returns scrolled points", async () => {
    const m = mockFetchSequence([
      {
        body: {
          result: { points: [{ id: 1, payload: { x: 1 } }] },
        },
      },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      const out = await s.list({ user_id: "u" });
      expect(out).toEqual([{ id: "1", payload: { x: 1 } }]);
    } finally {
      m.dispose();
    }
  });

  test("list defaults result and payload to empty", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      const out = await s.list();
      expect(out).toEqual([]);
    } finally {
      m.dispose();
    }
  });

  test("list with payload-less point uses empty payload", async () => {
    const m = mockFetchSequence([
      { body: { result: { points: [{ id: 5 }] } } },
    ]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      const out = await s.list();
      expect(out).toEqual([{ id: "5", payload: {} }]);
    } finally {
      m.dispose();
    }
  });

  test("deleteCollection issues DELETE", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      await s.deleteCollection();
      expect(m.calls[0]!.method).toBe("DELETE");
    } finally {
      m.dispose();
    }
  });

  test("non-ok response throws", async () => {
    const m = mockFetchSequence([{ status: 500, bodyText: "boom" }]);
    try {
      const s = new QdrantVectorStore({
        url: "http://q.test",
        collectionName: "col",
        dimension: 2,
      });
      await expect(s.deleteCollection()).rejects.toThrow(/Qdrant DELETE/);
    } finally {
      m.dispose();
    }
  });

  test("uses default url when omitted", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const s = new QdrantVectorStore({
        collectionName: "col",
        dimension: 2,
      });
      await s.delete("a");
      expect(m.calls[0]!.url).toBe(
        "http://localhost:6333/collections/col/points/delete?wait=true",
      );
    } finally {
      m.dispose();
    }
  });

  test("uses default collection name when omitted", async () => {
    const m = mockFetchSequence([{ body: {} }]);
    try {
      const s = new QdrantVectorStore({ dimension: 2 });
      await s.delete("a");
      expect(m.calls[0]!.url).toContain("/collections/mem0_default/");
    } finally {
      m.dispose();
    }
  });

  test("search result with missing payload uses empty object", async () => {
    const m = mockFetchSequence([
      { body: { result: [{ id: 1, score: 0.5 }] } },
    ]);
    try {
      const s = new QdrantVectorStore({
        dimension: 2,
        collectionName: "c",
      });
      const out = await s.search([0, 1], 1);
      expect(out).toEqual([
        { id: "1", payload: {}, score: 0.5 },
      ]);
    } finally {
      m.dispose();
    }
  });
});
