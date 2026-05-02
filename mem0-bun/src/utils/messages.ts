import type { Message } from "../types";

/**
 * Normalize messages input: a string becomes a single user message.
 * Validates structure; throws on bad input.
 */
export function normalizeMessages(input: string | Message[]): Message[] {
  if (input === undefined || input === null) {
    throw new Error("messages is required");
  }
  if (typeof input === "string") {
    return [{ role: "user", content: input }];
  }
  if (!Array.isArray(input)) {
    throw new Error("messages must be a string or an array of messages");
  }
  for (const m of input) {
    if (!m || typeof m !== "object") {
      throw new Error("each message must be an object with role and content");
    }
    if (typeof m.role !== "string" || typeof m.content !== "string") {
      throw new Error("each message must have string role and string content");
    }
  }
  return input;
}

/**
 * Render messages into a single user-readable transcript.
 * System messages are excluded so the LLM extractor doesn't pick up
 * instructions as facts.
 */
export function renderTranscript(messages: Message[]): string {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => `${m.role}: ${m.content}`)
    .join("\n");
}

const ENTITY_PARAMS = [
  "user_id",
  "agent_id",
  "run_id",
  "userId",
  "agentId",
  "runId",
];

export function rejectTopLevelEntityParams(
  config: Record<string, unknown>,
  methodName: string,
): void {
  const invalid = Object.keys(config).filter((k) => ENTITY_PARAMS.includes(k));
  if (invalid.length > 0) {
    throw new Error(
      `Top-level entity parameters [${invalid.join(", ")}] are not supported in ${methodName}(). Use filters: { user_id: "..." } instead.`,
    );
  }
}

export function validateEntityId(
  value: string | undefined,
  name: string,
): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string") {
    throw new Error(`Invalid ${name}: must be a string`);
  }
  const trimmed = value.trim();
  if (trimmed === "") {
    throw new Error(`Invalid ${name}: cannot be empty or whitespace-only`);
  }
  if (/\s/.test(trimmed)) {
    throw new Error(`Invalid ${name}: cannot contain whitespace`);
  }
  return trimmed;
}

export function validateSearchParams(
  threshold?: number,
  topK?: number,
): void {
  if (threshold !== undefined) {
    if (typeof threshold !== "number" || Number.isNaN(threshold)) {
      throw new Error("threshold must be a valid number");
    }
    if (threshold < 0 || threshold > 1) {
      throw new Error(
        `Invalid threshold: ${threshold}. Must be between 0 and 1`,
      );
    }
  }
  if (topK !== undefined) {
    if (
      typeof topK !== "number" ||
      Number.isNaN(topK) ||
      !Number.isInteger(topK)
    ) {
      throw new Error("topK must be a valid integer");
    }
    if (topK < 0) {
      throw new Error(`Invalid topK: ${topK}. Must be non-negative`);
    }
  }
}
