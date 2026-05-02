export const SCOPED_KEYS = ["user_id", "agent_id", "run_id"] as const;
export type ScopedKey = (typeof SCOPED_KEYS)[number];

/**
 * Build a parameterized WHERE clause for the conventional `user_id` /
 * `agent_id` / `run_id` columns. Returns an empty SQL string when no
 * scoped filters are defined.
 */
export function buildScopedWhere(filters?: Record<string, unknown>): {
  sql: string;
  params: (string | number | null)[];
} {
  const clauses: string[] = [];
  const params: (string | number | null)[] = [];
  if (filters) {
    for (const key of SCOPED_KEYS) {
      const v = filters[key];
      if (v === undefined) continue;
      clauses.push(`${key} = ?`);
      params.push(typeof v === "number" ? v : String(v));
    }
  }
  return {
    sql: clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "",
    params,
  };
}

/**
 * Match remaining payload filters in TS — used after SQL has already
 * scoped by `user_id`/`agent_id`/`run_id`. Skips undefined values and
 * the scoped keys themselves.
 */
export function matchPayloadFilters(
  payload: Record<string, unknown>,
  filters?: Record<string, unknown>,
): boolean {
  if (!filters) return true;
  for (const [k, v] of Object.entries(filters)) {
    if (v === undefined) continue;
    if (k === "user_id" || k === "agent_id" || k === "run_id") continue;
    if (payload[k] !== v) return false;
  }
  return true;
}

/** Returns true when `filters` carries any non-scoped key. */
export function hasExtraFilters(filters?: Record<string, unknown>): boolean {
  if (!filters) return false;
  for (const [k, v] of Object.entries(filters)) {
    if (v === undefined) continue;
    if (k === "user_id" || k === "agent_id" || k === "run_id") continue;
    return true;
  }
  return false;
}

/**
 * Pull the scoped-id columns out of a payload, preserving falsy non-null
 * values (`0`, `""`) instead of truthiness-coercing them to null.
 */
export function extractScopedColumns(
  payload: Record<string, unknown>,
): Record<ScopedKey, string | null> {
  const out = {
    user_id: null as string | null,
    agent_id: null as string | null,
    run_id: null as string | null,
  };
  for (const key of SCOPED_KEYS) {
    const v = payload[key];
    if (v != null) out[key] = String(v);
  }
  return out;
}
