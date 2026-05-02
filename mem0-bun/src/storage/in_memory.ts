import { HistoryManager, type HistoryAddInput } from "./base";
import { nowIso } from "../utils/hash";
import type { HistoryRecord } from "../types";

/**
 * Pure in-memory history manager used when SQLite is unavailable
 * or when callers explicitly want ephemeral history (e.g. edge runtimes).
 */
export class InMemoryHistoryManager extends HistoryManager {
  private records: HistoryRecord[] = [];
  private nextId = 1;

  constructor() {
    super();
  }

  async addHistory(input: HistoryAddInput): Promise<void> {
    this.records.push({
      id: this.nextId++,
      memoryId: input.memoryId,
      previousValue: input.previousValue,
      newValue: input.newValue,
      action: input.action,
      createdAt: input.createdAt ?? nowIso(),
      updatedAt: input.updatedAt ?? null,
      isDeleted: input.isDeleted ?? 0,
    });
  }

  async getHistory(memoryId: string): Promise<HistoryRecord[]> {
    return this.records
      .filter((r) => r.memoryId === memoryId)
      .map((r) => ({ ...r }));
  }

  async reset(): Promise<void> {
    this.records = [];
    this.nextId = 1;
  }

  async close(): Promise<void> {
    this.records = [];
  }
}
