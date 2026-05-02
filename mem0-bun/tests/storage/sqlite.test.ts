import { describe, expect, test } from "bun:test";
import { SqliteHistoryManager } from "../../src/storage/sqlite";

describe("SqliteHistoryManager", () => {
  test("add and read history with explicit timestamps", async () => {
    const h = new SqliteHistoryManager();
    await h.addHistory({
      memoryId: "m1",
      previousValue: null,
      newValue: "hello",
      action: "ADD",
      createdAt: "2025-01-01T00:00:00Z",
    });
    await h.addHistory({
      memoryId: "m1",
      previousValue: "hello",
      newValue: "hi",
      action: "UPDATE",
      createdAt: "2025-01-01T00:00:00Z",
      updatedAt: "2025-01-02T00:00:00Z",
    });
    await h.addHistory({
      memoryId: "m1",
      previousValue: "hi",
      newValue: null,
      action: "DELETE",
      isDeleted: 1,
    });

    const rows = await h.getHistory("m1");
    expect(rows).toHaveLength(3);
    expect(rows[0]!.action).toBe("ADD");
    expect(rows[2]!.isDeleted).toBe(1);
    expect(rows[1]!.updatedAt).toBe("2025-01-02T00:00:00Z");

    const empty = await h.getHistory("ghost");
    expect(empty).toHaveLength(0);
    await h.close();
  });

  test("default createdAt and isDeleted are filled in", async () => {
    const h = new SqliteHistoryManager();
    await h.addHistory({
      memoryId: "m2",
      previousValue: null,
      newValue: "x",
      action: "ADD",
    });
    const [row] = await h.getHistory("m2");
    expect(typeof row!.createdAt).toBe("string");
    expect(row!.isDeleted).toBe(0);
    await h.close();
  });

  test("reset clears all rows", async () => {
    const h = new SqliteHistoryManager();
    await h.addHistory({
      memoryId: "m3",
      previousValue: null,
      newValue: "x",
      action: "ADD",
    });
    await h.reset();
    expect(await h.getHistory("m3")).toHaveLength(0);
    await h.close();
  });

  test("uses provided path config", async () => {
    const h = new SqliteHistoryManager({ path: ":memory:" });
    await h.addHistory({
      memoryId: "m4",
      previousValue: null,
      newValue: "y",
      action: "ADD",
    });
    expect(await h.getHistory("m4")).toHaveLength(1);
    await h.close();
  });
});
