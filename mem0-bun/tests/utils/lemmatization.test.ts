import { describe, expect, test } from "bun:test";
import {
  lemmatizeForBm25,
  stem,
  stopwordsSet,
  tokenize,
} from "../../src/utils/lemmatization";

describe("lemmatization", () => {
  test("tokenize lowercases and strips punctuation", () => {
    expect(tokenize("Hello, World! Running fast.")).toEqual([
      "hello",
      "world",
      "running",
      "fast",
    ]);
  });

  test("tokenize handles empty input", () => {
    expect(tokenize("")).toEqual([]);
  });

  test("stem reduces inflected forms", () => {
    expect(stem("running")).toBe("run");
    expect(stem("cars")).toBe("car");
    expect(stem("happiness")).toBe("happi");
  });

  test("lemmatizeForBm25 drops stopwords + 1-letter tokens, then stems", () => {
    const out = lemmatizeForBm25("I am running with the cars in the city");
    expect(out).not.toContain("i");
    expect(out).not.toContain("am");
    expect(out).not.toContain("the");
    expect(out).toContain("run");
    expect(out).toContain("car");
    expect(out).toContain("citi"); // Porter on "city"
  });

  test("lemmatizeForBm25 returns empty for empty input", () => {
    expect(lemmatizeForBm25("")).toEqual([]);
  });

  test("lemmatizeForBm25 preserves duplicates (TF needs them)", () => {
    const out = lemmatizeForBm25("dog dog dog");
    expect(out).toEqual(["dog", "dog", "dog"]);
  });

  test("stopwordsSet exposes a non-empty read-only set", () => {
    const sw = stopwordsSet();
    expect(sw.has("the")).toBe(true);
    expect(sw.has("zebra")).toBe(false);
    expect(sw.size).toBeGreaterThan(50);
  });
});
