import nlp from "compromise";

export const ENTITY_TYPES = [
  "person",
  "place",
  "organization",
  "date",
  "value",
  "topic",
  "email",
  "url",
  "phone",
  "acronym",
  "code",
  "cjk",
] as const;

export type EntityType = (typeof ENTITY_TYPES)[number];

export const VALID_ENTITY_TYPES: ReadonlySet<EntityType> = new Set(
  ENTITY_TYPES,
);

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

/**
 * Strip leading/trailing punctuation that compromise sometimes leaves
 * attached. Keep internal punctuation so multi-word entities like
 * "St. Louis" or domain-style code identifiers survive.
 */
export function normalizeEntityText(text: string): string {
  return text.replace(/^[\s\p{P}]+|[\s\p{P}]+$/gu, "");
}

/** Stable dedup key for an entity. */
export function entityKey(entity: { type: string; text: string }): string {
  return `${entity.type}::${entity.text.toLowerCase()}`;
}

/**
 * Normalize and dedup a list of entities by `(type, lowercased-text)`.
 * Preserves the first occurrence's casing.
 */
export function dedupEntities(items: Entity[]): Entity[] {
  const seen = new Set<string>();
  const out: Entity[] = [];
  for (const e of items) {
    const cleaned = normalizeEntityText(e.text);
    if (!cleaned) continue;
    const normalized: Entity = { text: cleaned, type: e.type };
    const key = entityKey(normalized);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(normalized);
  }
  return out;
}

// ---------------- regex augmentations ----------------

const RE_EMAIL = /\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b/g;
const RE_URL = /\bhttps?:\/\/[^\s<>"'()一-鿿]+/gi;
const RE_PHONE = /(?:(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4})/g;
// 3-6 letter acronyms in ALL CAPS (e.g. NASA, OpenAI not matched — that's fine).
const RE_ACRONYM = /\b[A-Z]{3,6}\b/g;
// Code-style identifiers: snake_case, camelCase, dotted paths, file paths.
const RE_CODE = /\b(?:[A-Za-z_][\w]*\.[A-Za-z_][\w.]*|[a-z]+[A-Z][\w]+|[a-z]+_[a-z][\w]*|[\w/\\-]+\.[a-z]{2,5})\b/g;
// CJK noun chunks: runs of 2+ CJK characters (Han, Hiragana, Katakana, Hangul).
// 1-character chunks are too noisy.
const RE_CJK = /[一-鿿㐀-䶿぀-ゟ゠-ヿ가-힯]{2,}/g;

function regexEntities(text: string): Entity[] {
  const items: Entity[] = [];
  for (const m of text.matchAll(RE_EMAIL)) items.push({ text: m[0], type: "email" });
  for (const m of text.matchAll(RE_URL)) items.push({ text: m[0], type: "url" });
  for (const m of text.matchAll(RE_PHONE)) {
    // Filter: must contain enough digits to be a real phone number.
    const digits = m[0].replace(/\D/g, "");
    if (digits.length >= 7 && digits.length <= 15) {
      items.push({ text: m[0].trim(), type: "phone" });
    }
  }
  for (const m of text.matchAll(RE_ACRONYM)) items.push({ text: m[0], type: "acronym" });
  for (const m of text.matchAll(RE_CODE)) items.push({ text: m[0], type: "code" });
  for (const m of text.matchAll(RE_CJK)) items.push({ text: m[0], type: "cjk" });
  return items;
}

// ---------------- core extractor ----------------

/**
 * Extract named entities from a free-text string. Combines compromise's
 * NLP-level NER (English-leaning) with a regex pass for emails, URLs,
 * phone numbers, acronyms, code identifiers, and CJK noun chunks.
 *
 * Returns deduplicated entities (case-insensitive on text + same type)
 * preserving the original casing of the first occurrence.
 */
export function extractEntities(text: string): Entity[] {
  if (!text || !text.trim()) return [];
  const items: Entity[] = [];

  // compromise pass — best-effort English NER. Wrapped in try because
  // some inputs (extremely long, malformed Unicode) can blow up internally.
  try {
    const doc = nlp(text) as CompromiseDoc;
    for (const t of doc.people().out("array")) {
      items.push({ text: t, type: "person" });
    }
    for (const t of doc.places().out("array")) {
      items.push({ text: t, type: "place" });
    }
    for (const t of doc.organizations().out("array")) {
      items.push({ text: t, type: "organization" });
    }
    for (const t of doc.match("#Date+").out("array")) {
      items.push({ text: t, type: "date" });
    }
    for (const t of doc.match("#Value+").out("array")) {
      items.push({ text: t, type: "value" });
    }
    for (const t of doc.topics().out("array")) {
      items.push({ text: t, type: "topic" });
    }
  } catch {
    // Fall through to regex-only output.
  }

  items.push(...regexEntities(text));
  return dedupEntities(items);
}

/** Vectorized variant — returns one entity list per input string. */
export function extractEntitiesBatch(texts: string[]): Entity[][] {
  return texts.map((t) => extractEntities(t));
}
