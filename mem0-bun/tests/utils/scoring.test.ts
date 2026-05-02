import { describe, expect, test } from "bun:test";
import {
  entityBoostFor,
  scoreAndRank,
  type Candidate,
} from "../../src/utils/scoring";

describe("entityBoostFor", () => {
  test("similarity below 0.5 returns 0", () => {
    expect(entityBoostFor(0.49, 1, 0.3)).toBe(0);
  });

  test("attenuates with many linked memories", () => {
    const single = entityBoostFor(0.9, 1, 0.3);
    const many = entityBoostFor(0.9, 100, 0.3);
    expect(many).toBeLessThan(single);
    expect(single).toBeCloseTo(0.9 * 0.3, 5);
  });

  test("treats numLinked < 1 as 1", () => {
    expect(entityBoostFor(0.9, 0, 0.3)).toBeCloseTo(0.9 * 0.3, 5);
  });
});

describe("scoreAndRank", () => {
  const candidates: Candidate[] = [
    { id: "a", score: 0.8, payload: { data: "a" } },
    { id: "b", score: 0.7, payload: { data: "b" } },
    { id: "c", score: 0.6, payload: { data: "c" } },
  ];

  test("default weights produce semantic-led ranking", () => {
    const out = scoreAndRank(
      candidates,
      new Map(),
      new Map(),
      new Map(),
      0,
      10,
    );
    expect(out.map((c) => c.id)).toEqual(["a", "b", "c"]);
  });

  test("BM25 scores can override semantic order", () => {
    const out = scoreAndRank(
      candidates,
      new Map([["c", 1.0]]),
      new Map(),
      new Map(),
      0,
      10,
    );
    expect(out[0]!.id).toBe("c");
  });

  test("entity boost can lift", () => {
    const out = scoreAndRank(
      candidates,
      new Map(),
      new Map([["b", 1.0]]),
      new Map(),
      0,
      10,
    );
    expect(out[0]!.id).toBe("b");
  });

  test("graph boost can lift", () => {
    const out = scoreAndRank(
      candidates,
      new Map(),
      new Map(),
      new Map([["c", 1.0]]),
      0,
      10,
    );
    expect(out[0]!.id).toBe("c");
  });

  test("threshold filters", () => {
    const out = scoreAndRank(
      candidates,
      new Map(),
      new Map(),
      new Map(),
      0.75,
      10,
    );
    expect(out.map((c) => c.id)).toEqual(["a"]);
  });

  test("topK truncates", () => {
    const out = scoreAndRank(
      candidates,
      new Map(),
      new Map(),
      new Map(),
      0,
      2,
    );
    expect(out.length).toBe(2);
  });

  test("custom weights apply", () => {
    const out = scoreAndRank(
      candidates,
      new Map([["b", 1]]),
      new Map(),
      new Map(),
      0,
      10,
      { semantic: 0, bm25: 1, entity: 0, graph: 0 },
    );
    expect(out[0]!.id).toBe("b");
  });

  test("components are exposed for debug", () => {
    const out = scoreAndRank(
      candidates,
      new Map([["a", 0.5]]),
      new Map([["a", 0.4]]),
      new Map([["a", 0.3]]),
      0,
      1,
    );
    expect(out[0]!.components).toEqual({
      semantic: 0.8,
      bm25: 0.5,
      entity: 0.4,
      graph: 0.3,
    });
  });
});
