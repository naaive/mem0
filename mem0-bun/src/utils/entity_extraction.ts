import nlp from "compromise";

export type EntityType =
  | "person"
  | "place"
  | "organization"
  | "date"
  | "value"
  | "topic";

export interface Entity {
  text: string;
  type: EntityType;
}

interface CompromiseDoc {
  out: (kind: "array") => string[];
  people: () => CompromiseDoc;
  places: () => CompromiseDoc;
  organizations: () => CompromiseDoc;
  match: (pattern: string) => CompromiseDoc;
  topics: () => CompromiseDoc;
}

function normalize(text: string): string {
  // Strip leading/trailing punctuation that compromise sometimes leaves
  // attached (e.g. "Alice." vs "Alice"). Keep internal punctuation intact
  // so multi-word entities like "St. Louis" survive.
  return text.replace(/^[\s\p{P}]+|[\s\p{P}]+$/gu, "");
}

function unique(items: Entity[]): Entity[] {
  const seen = new Set<string>();
  const out: Entity[] = [];
  for (const e of items) {
    const cleaned = normalize(e.text);
    if (!cleaned) continue;
    const key = `${e.type}::${cleaned.toLowerCase()}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ text: cleaned, type: e.type });
  }
  return out;
}

/**
 * Extract named entities from a free-text string using compromise.
 * Returns deduplicated entities (case-insensitive on text) preserving
 * the original casing of the first occurrence.
 */
export function extractEntities(text: string): Entity[] {
  if (!text || !text.trim()) return [];
  const doc = nlp(text) as CompromiseDoc;
  const items: Entity[] = [];
  for (const t of doc.people().out("array")) items.push({ text: t, type: "person" });
  for (const t of doc.places().out("array")) items.push({ text: t, type: "place" });
  for (const t of doc.organizations().out("array")) {
    items.push({ text: t, type: "organization" });
  }
  for (const t of doc.match("#Date+").out("array")) {
    items.push({ text: t, type: "date" });
  }
  for (const t of doc.match("#Value+").out("array")) {
    items.push({ text: t, type: "value" });
  }
  for (const t of doc.topics().out("array")) items.push({ text: t, type: "topic" });
  return unique(items);
}

/** Vectorized variant — returns one entity list per input string. */
export function extractEntitiesBatch(texts: string[]): Entity[][] {
  return texts.map((t) => extractEntities(t));
}
