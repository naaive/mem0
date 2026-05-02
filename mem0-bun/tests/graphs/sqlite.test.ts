import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { SqliteGraphStore } from "../../src/graphs/sqlite";
import type { Triple } from "../../src/graphs/base";

const dim = 8;

function vecOf(text: string): number[] {
  const v = new Array<number>(dim).fill(0);
  for (let i = 0; i < text.length; i++) {
    v[text.charCodeAt(i) % dim]! += 1;
  }
  return v;
}
const embed = async (t: string): Promise<number[]> => vecOf(t);

const trip = (
  subject: string,
  relation: string,
  object: string,
  filters: Record<string, unknown>,
  memoryId?: string,
): Triple => ({ subject, relation, object, filters, memoryId });

describe("SqliteGraphStore", () => {
  const tmpDirs: string[] = [];
  afterEach(() => {
    for (const d of tmpDirs.splice(0)) {
      try {
        rmSync(d, { recursive: true, force: true });
      } catch {
        /* ignore */
      }
    }
  });

  test("initialize is a no-op (schema set up in ctor)", async () => {
    const g = new SqliteGraphStore();
    await g.initialize();
    expect(g.size()).toBe(0);
    g.close();
  });

  test("addTriples stores rows with scoped columns", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [
        trip("Alice", "knows", "Bob", { user_id: "u1" }, "m1"),
        trip("Bob", "lives_in", "Paris", { user_id: "u1" }, "m2"),
      ],
      embed,
    );
    expect(g.size()).toBe(2);
    g.close();
  });

  test("disk path persists triples across instances", async () => {
    const dir = mkdtempSync(join(tmpdir(), "mem0-bun-graph-"));
    tmpDirs.push(dir);
    const path = join(dir, "g.db");
    {
      const g = new SqliteGraphStore({ path });
      await g.addTriples(
        [trip("Alice", "knows", "Bob", { user_id: "u1" }, "m1")],
        embed,
      );
      g.close();
    }
    {
      const g = new SqliteGraphStore({ path });
      expect(g.size()).toBe(1);
      g.close();
    }
  });

  test("searchByEntities returns filter-scoped matches", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [
        trip("Alice", "lives_in", "Paris", { user_id: "u1" }, "m1"),
        trip("Bob", "knows", "Alice", { user_id: "u1" }, "m2"),
        trip("Charlie", "works_at", "Acme", { user_id: "u2" }, "m3"),
      ],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u1" } },
    );
    expect(out.length).toBeGreaterThan(0);
    expect(out.every((m) => m.filters.user_id === "u1")).toBe(true);
    g.close();
  });

  test("searchByEntities returns empty when no entities given", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [trip("A", "r", "B", { user_id: "u" }, "m")],
      embed,
    );
    expect(await g.searchByEntities([], { filters: { user_id: "u" } })).toEqual(
      [],
    );
    g.close();
  });

  test("returns empty when no rows match scope", async () => {
    const g = new SqliteGraphStore();
    expect(
      await g.searchByEntities(
        [{ text: "A", vector: vecOf("A") }],
        { filters: { user_id: "ghost" } },
      ),
    ).toEqual([]);
    g.close();
  });

  test("returns empty when no candidate passes similarity threshold", async () => {
    const g = new SqliteGraphStore();
    const orth = async (text: string) => {
      const v = new Array<number>(dim).fill(0);
      if (text === "alpha") v[0] = 1;
      else if (text === "beta") v[1] = 1;
      else if (text === "gamma") v[2] = 1;
      return v;
    };
    await g.addTriples(
      [trip("alpha", "r", "beta", { user_id: "u" }, "m1")],
      orth,
    );
    const out = await g.searchByEntities(
      [{ text: "gamma", vector: await orth("gamma") }],
      { filters: { user_id: "u" } },
    );
    expect(out).toEqual([]);
    g.close();
  });

  test("respects limit and applies hop attenuation", async () => {
    const g = new SqliteGraphStore();
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
      { filters: { user_id: "u" }, limit: 2, hops: 3 },
    );
    expect(out.length).toBeLessThanOrEqual(2);
    g.close();
  });

  test("respects extra payload filters via storedFilters", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [
        trip("Alice", "r", "X", { user_id: "u", tag: "a" }, "m1"),
        trip("Alice", "r", "Y", { user_id: "u", tag: "b" }, "m2"),
      ],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u", tag: "a" } },
    );
    expect(out.every((m) => m.filters.tag === "a")).toBe(true);
    g.close();
  });

  test("filter values that are undefined are skipped", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [trip("Alice", "r", "B", { user_id: "u" }, "m")],
      embed,
    );
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u", agent_id: undefined } },
    );
    expect(out.length).toBeGreaterThan(0);
    g.close();
  });

  test("triples with empty entity name are kept but not indexed for hops", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [trip("", "r", "Pizza", { user_id: "u" }, "m1")],
      embed,
    );
    expect(g.size()).toBe(1);
    g.close();
  });

  test("deleteByMemoryId scopes by filters", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [
        trip("A", "r", "B", { user_id: "u1" }, "m1"),
        trip("A", "r", "C", { user_id: "u1" }, "m2"),
        trip("A", "r", "D", { user_id: "u2" }, "m1"),
      ],
      embed,
    );
    await g.deleteByMemoryId("m1", { user_id: "u1" });
    expect(g.size()).toBe(2); // m2(u1) + m1(u2) survive
    g.close();
  });

  test("deleteByMemoryId works when no scope filters are supplied", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [
        trip("A", "r", "B", { user_id: "u1" }, "m1"),
        trip("A", "r", "C", { user_id: "u2" }, "m1"),
      ],
      embed,
    );
    await g.deleteByMemoryId("m1", {});
    expect(g.size()).toBe(0);
    g.close();
  });

  test("reset wipes triples + cache", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples([trip("A", "r", "B", { user_id: "u" }, "m")], embed);
    await g.reset();
    expect(g.size()).toBe(0);
    g.close();
  });

  test("hop expansion ends when no further peers found", async () => {
    const g = new SqliteGraphStore();
    await g.addTriples(
      [trip("Alice", "knows", "Bob", { user_id: "u" }, "m1")],
      embed,
    );
    // Request 5 hops on a single-edge graph; loop must terminate gracefully.
    const out = await g.searchByEntities(
      [{ text: "Alice", vector: vecOf("Alice") }],
      { filters: { user_id: "u" }, hops: 5 },
    );
    expect(out.length).toBe(1);
    g.close();
  });

  test("size returns 0 on freshly reset store", async () => {
    const g = new SqliteGraphStore();
    await g.reset();
    expect(g.size()).toBe(0);
    g.close();
  });
});
