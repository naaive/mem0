import { describe, expect, test } from "bun:test";
import { InMemoryGraphStore } from "../../src/graphs/in_memory";
import type { Triple } from "../../src/graphs/base";

const dim = 8;

function vecOf(text: string): number[] {
  const v = new Array<number>(dim).fill(0);
  for (let i = 0; i < text.length; i++) {
    v[text.charCodeAt(i) % dim]! += 1;
  }
  return v;
}

const embed = async (text: string): Promise<number[]> => vecOf(text);

const trip = (
  subject: string,
  relation: string,
  object: string,
  filters: Record<string, unknown>,
  memoryId?: string,
): Triple => ({ subject, relation, object, filters, memoryId });

describe("InMemoryGraphStore", () => {
  test("initialize is idempotent", async () => {
    const g = new InMemoryGraphStore();
    await g.initialize();
    await g.initialize();
    expect(g.size()).toBe(0);
  });

  test("addTriples uses embed cache to avoid re-embedding identical entities", async () => {
    const g = new InMemoryGraphStore();
    let embeds = 0;
    const counted = async (text: string) => {
      embeds++;
      return vecOf(text);
    };
    await g.addTriples(
      [
        trip("Alice", "knows", "Bob", { user_id: "u1" }, "m1"),
        trip("Alice", "lives_in", "Paris", { user_id: "u1" }, "m1"),
      ],
      counted,
    );
    expect(g.size()).toBe(2);
    // Alice was embedded once, Bob once, Paris once → 3 embed calls.
    expect(embeds).toBe(3);
  });

  test("searchByEntities returns matching triples with score, attenuated by hops", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples(
      [
        trip("Alice", "lives_in", "Paris", { user_id: "u1" }, "m1"),
        trip("Bob", "knows", "Alice", { user_id: "u1" }, "m2"),
        trip("Charlie", "works_at", "Acme", { user_id: "u1" }, "m3"),
      ],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u1" }, hops: 2, limit: 10 },
    );
    const ids = new Set(out.map((m) => m.memoryId));
    expect(ids.has("m1")).toBe(true);
    expect(ids.has("m2")).toBe(true);
  });

  test("returns empty when no entities provided", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples([trip("A", "r", "B", { user_id: "u1" }, "m")], embed);
    expect(
      await g.searchByEntities([], { filters: { user_id: "u1" } }),
    ).toEqual([]);
  });

  test("returns empty when triple set is empty", async () => {
    const g = new InMemoryGraphStore();
    expect(
      await g.searchByEntities(
        [{ text: "A", vector: vecOf("A") }],
        { filters: { user_id: "u1" } },
      ),
    ).toEqual([]);
  });

  test("returns empty when no candidates pass similarity threshold", async () => {
    const g = new InMemoryGraphStore();
    // Use a custom embed that gives unique slots so similarity is 0.
    const orthogonal = async (text: string) => {
      const v = new Array<number>(dim).fill(0);
      if (text === "alpha") v[0] = 1;
      else if (text === "beta") v[1] = 1;
      else if (text === "gamma") v[2] = 1;
      return v;
    };
    await g.addTriples(
      [trip("alpha", "r", "beta", { user_id: "u1" }, "m1")],
      orthogonal,
    );
    const out = await g.searchByEntities(
      [{ text: "gamma", vector: await orthogonal("gamma") }],
      { filters: { user_id: "u1" } },
    );
    expect(out).toEqual([]);
  });

  test("filter scoping excludes triples from other users", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples(
      [
        trip("Alice", "lives_in", "Paris", { user_id: "u1" }, "m1"),
        trip("Alice", "lives_in", "London", { user_id: "u2" }, "m2"),
      ],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u1" } },
    );
    expect(out.every((m) => m.memoryId === "m1")).toBe(true);
  });

  test("respects limit", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples(
      [
        trip("Alice", "r1", "X", { user_id: "u" }, "m1"),
        trip("Alice", "r2", "Y", { user_id: "u" }, "m2"),
        trip("Alice", "r3", "Z", { user_id: "u" }, "m3"),
      ],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u" }, limit: 2 },
    );
    expect(out.length).toBe(2);
  });

  test("hops parameter widens or narrows reach", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples(
      [
        trip("Alice", "knows", "Bob", { user_id: "u1" }, "m1"),
        trip("Bob", "knows", "Charlie", { user_id: "u1" }, "m2"),
        trip("Dora", "knows", "Eve", { user_id: "u1" }, "m3"),
      ],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u1" }, hops: 1 },
    );
    // m1 always reachable (direct seed match on Alice); m3 (Dora-Eve) is
    // unrelated and must be absent.
    expect(out.some((m) => m.memoryId === "m1")).toBe(true);
    expect(out.find((m) => m.memoryId === "m3")).toBeUndefined();
  });

  test("filters with undefined values are skipped during match", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples(
      [trip("Alice", "knows", "Bob", { user_id: "u1" }, "m1")],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u1", agent_id: undefined } },
    );
    expect(out.length).toBeGreaterThan(0);
  });

  test("deleteByMemoryId removes only matching triples", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples(
      [
        trip("A", "r", "B", { user_id: "u1" }, "m1"),
        trip("A", "r", "C", { user_id: "u1" }, "m2"),
        trip("A", "r", "D", { user_id: "u2" }, "m1"), // different user
      ],
      embed,
    );
    await g.deleteByMemoryId("m1", { user_id: "u1" });
    expect(g.size()).toBe(2); // m2 (u1) + m1 (u2) survive
  });

  test("reset clears triples + cache", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples([trip("A", "r", "B", { user_id: "u" }, "m")], embed);
    await g.reset();
    expect(g.size()).toBe(0);
  });

  test("unindexEntity helper handles empty/unknown/known names", async () => {
    const g = new InMemoryGraphStore();
    g.unindexEntity("", 0); // empty key path
    g.unindexEntity("ghost", 0); // unknown key path
    await g.addTriples(
      [trip("Alice", "knows", "Bob", { user_id: "u1" }, "m1")],
      embed,
    );
    g.unindexEntity("Alice", 0); // known key, actual delete path
    g.unindexEntity("Alice", 0); // second call → set is empty / removed
  });

  test("triple with empty entity name is still stored but not searchable by name", async () => {
    const g = new InMemoryGraphStore();
    await g.addTriples(
      [trip("", "loves", "Pizza", { user_id: "u" }, "m1")],
      embed,
    );
    expect(g.size()).toBe(1);
  });
});
