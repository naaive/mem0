import { describe, expect, test } from "bun:test";
import {
  normalizeMessages,
  rejectTopLevelEntityParams,
  renderTranscript,
  validateEntityId,
  validateSearchParams,
} from "../../src/utils/messages";

describe("normalizeMessages", () => {
  test("string becomes single user message", () => {
    expect(normalizeMessages("hi")).toEqual([{ role: "user", content: "hi" }]);
  });
  test("passes through array messages", () => {
    const m = [{ role: "user", content: "a" }];
    expect(normalizeMessages(m)).toBe(m);
  });
  test("rejects null", () => {
    expect(() => normalizeMessages(null as unknown as string)).toThrow(
      /required/,
    );
  });
  test("rejects non-array, non-string", () => {
    expect(() =>
      normalizeMessages(42 as unknown as string),
    ).toThrow(/string or an array/);
  });
  test("rejects non-object element", () => {
    expect(() =>
      normalizeMessages([null as unknown as { role: string; content: string }]),
    ).toThrow(/each message/);
  });
  test("rejects bad role/content types", () => {
    expect(() =>
      normalizeMessages([
        { role: 1 as unknown as string, content: "x" },
      ]),
    ).toThrow(/string role/);
  });
});

describe("renderTranscript", () => {
  test("excludes system messages", () => {
    const out = renderTranscript([
      { role: "system", content: "ignore me" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ]);
    expect(out).toBe("user: hi\nassistant: hello");
  });
});

describe("rejectTopLevelEntityParams", () => {
  test("noop when no entity keys", () => {
    rejectTopLevelEntityParams({ topK: 5 }, "search");
  });
  test("throws when entity key present", () => {
    expect(() =>
      rejectTopLevelEntityParams({ user_id: "u1" }, "search"),
    ).toThrow(/Top-level entity/);
  });
});

describe("validateEntityId", () => {
  test("undefined returns undefined", () => {
    expect(validateEntityId(undefined, "user")).toBeUndefined();
  });
  test("trims whitespace", () => {
    expect(validateEntityId("  alice  ", "user")).toBe("alice");
  });
  test("rejects non-string", () => {
    expect(() =>
      validateEntityId(42 as unknown as string, "user"),
    ).toThrow(/must be a string/);
  });
  test("rejects empty", () => {
    expect(() => validateEntityId("   ", "user")).toThrow(/empty/);
  });
  test("rejects internal whitespace", () => {
    expect(() => validateEntityId("a b", "user")).toThrow(/whitespace/);
  });
});

describe("validateSearchParams", () => {
  test("accepts undefined", () => {
    validateSearchParams();
  });
  test("rejects NaN threshold", () => {
    expect(() => validateSearchParams(NaN)).toThrow(/valid number/);
  });
  test("rejects non-number threshold", () => {
    expect(() =>
      validateSearchParams("0.5" as unknown as number),
    ).toThrow(/valid number/);
  });
  test("rejects threshold below 0", () => {
    expect(() => validateSearchParams(-0.1)).toThrow(/between 0 and 1/);
  });
  test("rejects threshold above 1", () => {
    expect(() => validateSearchParams(1.5)).toThrow(/between 0 and 1/);
  });
  test("rejects NaN topK", () => {
    expect(() => validateSearchParams(undefined, NaN)).toThrow(
      /valid integer/,
    );
  });
  test("rejects non-integer topK", () => {
    expect(() => validateSearchParams(undefined, 1.5)).toThrow(
      /valid integer/,
    );
  });
  test("rejects non-number topK", () => {
    expect(() =>
      validateSearchParams(undefined, "5" as unknown as number),
    ).toThrow(/valid integer/);
  });
  test("rejects negative topK", () => {
    expect(() => validateSearchParams(undefined, -1)).toThrow(
      /non-negative/,
    );
  });
  test("accepts valid", () => {
    validateSearchParams(0.5, 10);
  });
});
