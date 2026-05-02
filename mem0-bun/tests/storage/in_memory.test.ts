import { describe, expect, test } from "bun:test";
import { InMemoryHistoryManager } from "../../src/storage/in_memory";

describe("InMemoryHistoryManager", () => {
  test("CRUD-ish flow", async () => {
    const h = new InMemoryHistoryManager();
    await h.addHistory({
      memoryId: "m1",
      previousValue: null,
      newValue: "x",
      action: "ADD",
      createdAt: "2025-01-01T00:00:00Z",
    });
    await h.addHistory({
      memoryId: "m1",
      previousValue: "x",
      newValue: "y",
      action: "UPDATE",
    });
    const rows = await h.getHistory("m1");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.id).toBe(1);
    expect(rows[1]!.id).toBe(2);
    expect(rows[1]!.updatedAt).toBeNull();
    expect(rows[1]!.isDeleted).toBe(0);
    expect(typeof rows[1]!.createdAt).toBe("string");

    expect(await h.getHistory("missing")).toEqual([]);

    await h.reset();
    expect(await h.getHistory("m1")).toHaveLength(0);
    await h.addHistory({
      memoryId: "m2",
      previousValue: null,
      newValue: "z",
      action: "ADD",
    });
    expect((await h.getHistory("m2"))[0]!.id).toBe(1);

    await h.close();
    expect(await h.getHistory("m2")).toHaveLength(0);
  });
});
