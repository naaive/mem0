import { describe, expect, test } from "bun:test";
import {
  buildCorpusStats,
  getBm25Params,
  normalizeBm25,
  scoreBm25,
} from "../../src/utils/bm25";

describe("buildCorpusStats", () => {
  test("computes docCount, avgDocLen, df", () => {
    const stats = buildCorpusStats([
      ["a", "b", "c"],
      ["a", "a", "d"],
      ["e"],
    ]);
    expect(stats.docCount).toBe(3);
    expect(stats.avgDocLen).toBeCloseTo(7 / 3);
    expect(stats.df.get("a")).toBe(2);
    expect(stats.df.get("b")).toBe(1);
    expect(stats.df.get("e")).toBe(1);
  });

  test("returns 0 avgDocLen for empty corpus", () => {
    const stats = buildCorpusStats([]);
    expect(stats.docCount).toBe(0);
    expect(stats.avgDocLen).toBe(0);
  });
});

describe("scoreBm25", () => {
  test("returns 0 when corpus empty", () => {
    const stats = buildCorpusStats([]);
    expect(scoreBm25(["x"], ["x"], stats)).toBe(0);
  });

  test("returns 0 when doc empty", () => {
    const stats = buildCorpusStats([["a"], ["b"]]);
    expect(scoreBm25(["a"], [], stats)).toBe(0);
  });

  test("returns 0 when no query terms appear", () => {
    const stats = buildCorpusStats([["a", "b"]]);
    expect(scoreBm25(["c"], ["a", "b"], stats)).toBe(0);
  });

  test("ranks document with all query terms higher than partial match", () => {
    const docs = [
      ["pizza", "pepperoni"],
      ["pizza"],
      ["broccoli"],
    ];
    const stats = buildCorpusStats(docs);
    const full = scoreBm25(["pizza", "pepperoni"], docs[0]!, stats);
    const partial = scoreBm25(["pizza", "pepperoni"], docs[1]!, stats);
    const none = scoreBm25(["pizza", "pepperoni"], docs[2]!, stats);
    expect(full).toBeGreaterThan(partial);
    expect(partial).toBeGreaterThan(none);
  });

  test("respects k1 and b options", () => {
    const docs = [["a", "b", "a"], ["b"]];
    const stats = buildCorpusStats(docs);
    const default_score = scoreBm25(["a"], docs[0]!, stats);
    const high_k1 = scoreBm25(["a"], docs[0]!, stats, { k1: 5, b: 0.75 });
    expect(high_k1).not.toBe(default_score);
  });

  test("handles avgDocLen = 0 path safely", () => {
    // Force avgDocLen to 0 explicitly.
    const stats = { docCount: 1, avgDocLen: 0, df: new Map([["a", 1]]) };
    const score = scoreBm25(["a"], ["a"], stats);
    expect(Number.isFinite(score)).toBe(true);
  });
});

describe("normalizeBm25", () => {
  test("returns 0/1 when steepness is 0", () => {
    expect(normalizeBm25(0.5, 1, 0)).toBe(0);
    expect(normalizeBm25(2, 1, 0)).toBe(1);
    expect(normalizeBm25(1, 1, 0)).toBe(1);
  });

  test("sigmoid output is monotonically increasing", () => {
    const a = normalizeBm25(2, 5, 1);
    const b = normalizeBm25(5, 5, 1);
    const c = normalizeBm25(8, 5, 1);
    expect(a).toBeLessThan(b);
    expect(b).toBeLessThan(c);
    expect(b).toBeCloseTo(0.5, 5);
  });
});

describe("getBm25Params", () => {
  test("very short queries", () => {
    expect(getBm25Params("hi", ["hi"])).toEqual([1.5, 1.5]);
  });
  test("short queries", () => {
    expect(getBm25Params("a b c", ["a", "b", "c"])).toEqual([3, 1.2]);
  });
  test("medium queries", () => {
    expect(getBm25Params("a b c d e", ["a", "b", "c", "d", "e"])).toEqual([
      5, 1.0,
    ]);
  });
  test("longer queries (>6 tokens, <=100 chars)", () => {
    expect(getBm25Params("a b c d e f g", ["a", "b", "c", "d", "e", "f", "g"]))
      .toEqual([6, 0.8]);
  });
  test("very long character count queries", () => {
    const long = "abc ".repeat(50);
    expect(getBm25Params(long, ["a", "b", "c", "d", "e", "f", "g"])).toEqual([
      8, 0.6,
    ]);
  });
});
