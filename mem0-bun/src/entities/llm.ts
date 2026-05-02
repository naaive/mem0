import type { LLM } from "../llms/base";
import { extractJson } from "../utils/json";
import type { EntityExtractor } from "./base";
import type { Entity, EntityType } from "../utils/entity_extraction";

const ENTITY_EXTRACTION_SYSTEM = `You extract named entities from short pieces of text. Recognize entities of any language including Chinese, Japanese, Korean, Arabic, Cyrillic, etc.

Allowed types:
- person (a named individual)
- place (city, country, landmark, region)
- organization (company, agency, team)
- date (any date or relative time expression)
- value (numeric value with optional unit, money, measurement)
- topic (a subject or concept the text is about)
- email
- url
- phone
- acronym (3-6 letter all-caps abbreviation)

Return STRICT JSON of the form: {"entities": [{"text": "...", "type": "..."}]}.
Do not invent entities not present in the text. If none, return {"entities": []}.`;

const VALID_TYPES: ReadonlySet<EntityType> = new Set([
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
]);

/**
 * LLM-driven entity extractor. Multi-language by construction. Slower and
 * more expensive than the local extractor; suitable for non-English text
 * or when the local pass returns nothing.
 *
 * Results are cached per-text (unbounded) so repeated calls are free.
 */
export class LLMEntityExtractor implements EntityExtractor {
  private readonly llm: LLM;
  private readonly cache = new Map<string, Entity[]>();

  constructor(llm: LLM) {
    this.llm = llm;
  }

  private parse(raw: string): Entity[] {
    let parsed: { entities?: unknown };
    try {
      parsed = JSON.parse(extractJson(raw));
    } catch {
      return [];
    }
    if (!parsed || !Array.isArray(parsed.entities)) return [];
    const out: Entity[] = [];
    const seen = new Set<string>();
    for (const item of parsed.entities as unknown[]) {
      if (typeof item !== "object" || item === null) continue;
      const r = item as Record<string, unknown>;
      const text = typeof r.text === "string" ? r.text.trim() : "";
      const typeRaw = typeof r.type === "string" ? r.type.toLowerCase() : "";
      if (!text || !VALID_TYPES.has(typeRaw as EntityType)) continue;
      const key = `${typeRaw}::${text.toLowerCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ text, type: typeRaw as EntityType });
    }
    return out;
  }

  async extract(text: string): Promise<Entity[]> {
    if (!text || !text.trim()) return [];
    const cacheKey = text.trim();
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;
    let raw: string;
    try {
      raw = await this.llm.generateResponse(
        [
          { role: "system", content: ENTITY_EXTRACTION_SYSTEM },
          { role: "user", content: text },
        ],
        { responseFormat: "json_object" },
      );
    } catch {
      return [];
    }
    const entities = this.parse(raw);
    this.cache.set(cacheKey, entities);
    return entities;
  }

  async extractBatch(texts: string[]): Promise<Entity[][]> {
    const out: Entity[][] = [];
    for (const t of texts) out.push(await this.extract(t));
    return out;
  }
}
