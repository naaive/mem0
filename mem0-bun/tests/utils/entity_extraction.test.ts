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
});
