import { Database } from "bun:sqlite";
import { HistoryManager, type HistoryAddInput } from "./base";
import { nowIso } from "../utils/hash";
import type { HistoryRecord, HistoryStoreConfig } from "../types";

/**
 * Local-first SQLite-backed history manager.
 * Defaults to in-memory (":memory:") so tests are hermetic; pass a path
 * via config for persistence.
 */
export class SqliteHistoryManager extends HistoryManager {
  private readonly db: Database;

  constructor(config: HistoryStoreConfig = {}) {
    super();
    this.db = new Database(config.path ?? ":memory:");
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id TEXT NOT NULL,
        previous_value TEXT,
        new_value TEXT,
        action TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT,
        is_deleted INTEGER NOT NULL DEFAULT 0
      );
      CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id);
    `);
  }

  async addHistory(input: HistoryAddInput): Promise<void> {
    const stmt = this.db.prepare(
      `INSERT INTO history (memory_id, previous_value, new_value, action, created_at, updated_at, is_deleted)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    );
    stmt.run(
      input.memoryId,
      input.previousValue,
      input.newValue,
      input.action,
      input.createdAt ?? nowIso(),
      input.updatedAt ?? null,
      input.isDeleted ?? 0,
    );
  }

  async getHistory(memoryId: string): Promise<HistoryRecord[]> {
    const rows = this.db
      .prepare(
        `SELECT id, memory_id as memoryId, previous_value as previousValue,
                new_value as newValue, action, created_at as createdAt,
                updated_at as updatedAt, is_deleted as isDeleted
         FROM history WHERE memory_id = ? ORDER BY id ASC`,
      )
      .all(memoryId) as HistoryRecord[];
    return rows;
  }

  async reset(): Promise<void> {
    this.db.exec("DELETE FROM history");
  }

  async close(): Promise<void> {
    this.db.close();
  }
}
