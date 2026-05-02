import { describe, expect, test } from "bun:test";
import { extractJson, safeJsonParse } from "../../src/utils/json";

describe("extractJson", () => {
  test("returns empty for empty input", () => {
    expect(extractJson("")).toBe("");
  });

  test("strips ```json fences", () => {
    const out = extractJson('```json\n{"a":1}\n```');
    expect(out).toBe('{"a":1}');
  });

  test("strips ``` fences without language", () => {
    const out = extractJson('```\n{"a":1}\n```');
    expect(out).toBe('{"a":1}');
  });

  test("handles preamble before object", () => {
    const out = extractJson('here is JSON: {"a":1} bye');
    expect(out).toBe('{"a":1}');
  });

  test("handles arrays", () => {
    const out = extractJson("prefix [1, 2, 3] suffix");
    expect(out).toBe("[1, 2, 3]");
  });

  test("handles braces inside strings", () => {
    const raw = '{"a": "has } brace"}';
    expect(extractJson(raw)).toBe(raw);
  });

  test("handles escaped quote inside string", () => {
    const raw = '{"a": "esc\\"aped \\\\ braces } "}';
    expect(extractJson(raw)).toBe(raw);
  });

  test("returns text without braces unchanged when no JSON-ish chars", () => {
    expect(extractJson("hello world")).toBe("hello world");
  });

  test("returns suffix when JSON is unbalanced", () => {
    const out = extractJson('garbage {"a":1');
    expect(out).toBe('{"a":1');
  });
});

describe("safeJsonParse", () => {
  test("parses valid json", () => {
    expect(safeJsonParse<{ a: number } | null>('{"a":1}', null)).toEqual({
      a: 1,
    });
  });
  test("returns fallback on invalid", () => {
    expect(safeJsonParse("not json", { fallback: true })).toEqual({
      fallback: true,
    });
  });
});
