import type { HistoryRecord } from "../types";

export interface HistoryAddInput {
  memoryId: string;
  previousValue: string | null;
  newValue: string | null;
  action: "ADD" | "UPDATE" | "DELETE";
  createdAt?: string;
  updatedAt?: string;
  isDeleted?: number;
}

export abstract class HistoryManager {
  constructor() {
    // Subclasses may override; explicit no-op keeps the base class
    // observable for instantiation checks.
  }
  abstract addHistory(input: HistoryAddInput): Promise<void>;
  abstract getHistory(memoryId: string): Promise<HistoryRecord[]>;
  abstract reset(): Promise<void>;
  abstract close(): Promise<void>;
}
