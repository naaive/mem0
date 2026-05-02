import type { LLM } from "../llms/base";
import { extractJson } from "../utils/json";
import type { EntityExtractor } from "./base";
import {
  VALID_ENTITY_TYPES,
  dedupEntities,
  type Entity,
  type EntityType,
} from "../utils/entity_extraction";

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

/**
 * LLM-driven entity extractor. Multi-language by construction. Slower and
 * more expensive than the local extractor; suitable for non-English text
 * or when the local pass returns nothing.
 *
 * Results are cached per-text with LRU eviction so a long-running process
 * with diverse inputs cannot leak memory.
 */
export class LLMEntityExtractor implements EntityExtractor {
  private readonly llm: LLM;
  private readonly cache = new Map<string, Entity[]>();
  private readonly cacheMax: number;

  constructor(llm: LLM, options: { cacheMax?: number } = {}) {
    this.llm = llm;
    this.cacheMax = options.cacheMax ?? 1024;
  }

  private cacheGet(key: string): Entity[] | undefined {
    const v = this.cache.get(key);
    if (v === undefined) return undefined;
    // Refresh LRU position.
    this.cache.delete(key);
    this.cache.set(key, v);
    return v;
  }

  private cacheSet(key: string, value: Entity[]): void {
    this.cache.set(key, value);
    if (this.cache.size > this.cacheMax) {
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }
  }

  /** Drop all cached results. */
  clearCache(): void {
    this.cache.clear();
  }

  private parse(raw: string): Entity[] {
    let parsed: { entities?: unknown };
    try {
      parsed = JSON.parse(extractJson(raw));
    } catch {
      return [];
    }
    if (!parsed || !Array.isArray(parsed.entities)) return [];
    const candidates: Entity[] = [];
    for (const item of parsed.entities as unknown[]) {
      if (typeof item !== "object" || item === null) continue;
      const r = item as Record<string, unknown>;
      const text = typeof r.text === "string" ? r.text : "";
      const typeRaw = typeof r.type === "string" ? r.type.toLowerCase() : "";
      if (!text.trim() || !VALID_ENTITY_TYPES.has(typeRaw as EntityType)) {
        continue;
      }
      candidates.push({ text, type: typeRaw as EntityType });
    }
    return dedupEntities(candidates);
  }

  async extract(text: string): Promise<Entity[]> {
    if (!text || !text.trim()) return [];
    const cacheKey = text.trim();
    const cached = this.cacheGet(cacheKey);
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
    this.cacheSet(cacheKey, entities);
    return entities;
  }

  async extractBatch(texts: string[]): Promise<Entity[][]> {
    return Promise.all(texts.map((t) => this.extract(t)));
  }
}
