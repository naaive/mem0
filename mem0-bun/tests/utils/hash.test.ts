import { describe, expect, test } from "bun:test";
import { md5, uuid, nowIso, cosineSimilarity } from "../../src/utils/hash";

describe("hash utils", () => {
  test("md5 returns 32-char hex", () => {
    const h = md5("hello");
    expect(h).toMatch(/^[a-f0-9]{32}$/);
    expect(h).toBe("5d41402abc4b2a76b9719d911017c592");
  });

  test("uuid is rfc4122-ish", () => {
    const id = uuid();
    expect(id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
    );
  });

  test("nowIso returns ISO date string", () => {
    const v = nowIso();
    expect(new Date(v).toISOString()).toBe(v);
  });

  test("cosineSimilarity returns 1 for identical vectors", () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 6);
  });

  test("cosineSimilarity returns 0 for orthogonal", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
  });

  test("cosineSimilarity returns 0 when length mismatch", () => {
    expect(cosineSimilarity([1, 2], [1, 2, 3])).toBe(0);
  });

  test("cosineSimilarity returns 0 on zero vectors", () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
    expect(cosineSimilarity([1, 1], [0, 0])).toBe(0);
  });

  test("cosineSimilarity returns 0 when both empty", () => {
    expect(cosineSimilarity([], [])).toBe(0);
  });
});
