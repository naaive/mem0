import { describe, expect, test } from "bun:test";
import {
  extractEntities,
  extractEntitiesBatch,
} from "../../src/utils/entity_extraction";

describe("entity_extraction", () => {
  test("extracts people, places, dates from a sentence", () => {
    const ents = extractEntities(
      "Alice met Bob in Paris on January 5th 2025.",
    );
    const labels = new Set(ents.map((e) => `${e.type}:${e.text.toLowerCase()}`));
    expect([...labels].some((l) => l.startsWith("person:alice"))).toBe(true);
    expect([...labels].some((l) => l.startsWith("person:bob"))).toBe(true);
    expect([...labels].some((l) => l.startsWith("place:paris"))).toBe(true);
    expect([...labels].some((l) => l.startsWith("date:"))).toBe(true);
  });

  test("returns empty array for empty/whitespace input", () => {
    expect(extractEntities("")).toEqual([]);
    expect(extractEntities("   ")).toEqual([]);
  });

  test("dedupes case-insensitive entities", () => {
    const ents = extractEntities("Alice and ALICE met Alice.");
    const persons = ents.filter((e) => e.type === "person");
    expect(persons.length).toBe(1);
  });

  test("extractEntitiesBatch returns one list per input", () => {
    const out = extractEntitiesBatch(["Alice", "Bob"]);
    expect(out.length).toBe(2);
    expect(out[0]!.length).toBeGreaterThan(0);
  });

  test("recognizes places when present", () => {
    const ents = extractEntities("Acme Corp launched a product in San Francisco.");
    expect(ents.some((e) => /san francisco/i.test(e.text))).toBe(true);
    // Smoke check: extractor is reachable for places-only text
    expect(ents.length).toBeGreaterThan(0);
  });

  test("handles a numeric-rich text producing value entities", () => {
    const ents = extractEntities("I bought 5 apples and 12 oranges.");
    expect(ents.some((e) => e.type === "value")).toBe(true);
  });

  test("extracts emails", () => {
    const ents = extractEntities("Email me at alice@example.com or bob@x.io.");
    const emails = ents.filter((e) => e.type === "email").map((e) => e.text);
    expect(emails).toContain("alice@example.com");
    expect(emails).toContain("bob@x.io");
  });

  test("extracts URLs", () => {
    const ents = extractEntities(
      "Check https://example.com/path and http://x.io",
    );
    const urls = ents.filter((e) => e.type === "url").map((e) => e.text);
    expect(urls.some((u) => u.startsWith("https://example.com"))).toBe(true);
    expect(urls.some((u) => u.startsWith("http://x.io"))).toBe(true);
  });

  test("extracts phone numbers", () => {
    const ents = extractEntities("Call me at +1 415 555 1234.");
    const phones = ents.filter((e) => e.type === "phone");
    expect(phones.length).toBeGreaterThan(0);
  });

  test("rejects too-short digit runs as phones", () => {
    const ents = extractEntities("Just 12 nothing else.");
    const phones = ents.filter((e) => e.type === "phone");
    expect(phones.length).toBe(0);
  });

  test("extracts ALL-CAPS acronyms (3-6 letters)", () => {
    const ents = extractEntities("NASA and FBI launched a JWT spec.");
    const acronyms = ents.filter((e) => e.type === "acronym").map((e) => e.text);
    expect(acronyms).toEqual(expect.arrayContaining(["NASA", "FBI", "JWT"]));
  });

  test("extracts code-style identifiers", () => {
    const ents = extractEntities(
      "Use snake_case_var or camelCaseVar in foo.bar.baz module, or src/index.ts file.",
    );
    const code = ents.filter((e) => e.type === "code").map((e) => e.text);
    expect(code.some((c) => /snake_case_var/.test(c))).toBe(true);
    expect(code.some((c) => /camelCaseVar/.test(c))).toBe(true);
    expect(code.some((c) => /foo\.bar/.test(c))).toBe(true);
    expect(code.some((c) => /src\/index\.ts/.test(c))).toBe(true);
  });

  test("extracts CJK noun chunks", () => {
    const ents = extractEntities("我喜欢吃北京烤鸭和宫保鸡丁。");
    const cjk = ents.filter((e) => e.type === "cjk").map((e) => e.text);
    expect(cjk.length).toBeGreaterThan(0);
    expect(cjk.some((c) => /北京/.test(c))).toBe(true);
  });

  test("extracts Japanese kana chunks", () => {
    const ents = extractEntities("こんにちは、東京で会いましょう。");
    const cjk = ents.filter((e) => e.type === "cjk");
    expect(cjk.length).toBeGreaterThan(0);
  });

  test("extracts Korean Hangul chunks", () => {
    const ents = extractEntities("안녕하세요 서울에서 만나요");
    const cjk = ents.filter((e) => e.type === "cjk");
    expect(cjk.length).toBeGreaterThan(0);
  });

  test("compromise crash is non-fatal (regex-only fallback)", () => {
    // We can't easily force compromise to crash on benign input; just ensure
    // the extractor doesn't throw for unusual unicode soup.
    const ents = extractEntities("\u{1F600} alpha-beta_gamma 1234567");
    expect(Array.isArray(ents)).toBe(true);
  });
});
