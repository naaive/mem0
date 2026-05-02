import type { EntityExtractor } from "./base";
import type { Entity } from "../utils/entity_extraction";
import { LocalEntityExtractor } from "./local";
import { LLMEntityExtractor } from "./llm";
import type { LLM } from "../llms/base";

export interface HybridEntityExtractorOptions {
  /**
   * If the local extractor returns fewer than this many entities, the LLM
   * fallback is invoked. Default 1 — i.e. fall back only when nothing was
   * found locally.
   */
  llmFallbackThreshold?: number;
  /**
   * Always invoke the LLM extractor when text contains non-Latin script
   * (CJK, Arabic, Cyrillic, etc). Default true. Cheaper than running the
   * LLM unconditionally while still capturing multi-language coverage.
   */
  llmForNonLatin?: boolean;
}

const NON_LATIN_RE = /[Ѐ-ӿ֐-׿؀-ۿ܀-ݏऀ-ॿ฀-๿　-ヿ一-鿿가-ퟯ]/u;

/**
 * Two-tier extractor. Tries local (compromise + regex) first; if the local
 * pass returns fewer than `llmFallbackThreshold` entities (default 1), or
 * the input contains non-Latin characters, falls back to the LLM extractor.
 *
 * Entities from both passes are unioned and deduplicated by `(type, text)`.
 */
export class HybridEntityExtractor implements EntityExtractor {
  private readonly local: LocalEntityExtractor;
  private readonly llm: LLMEntityExtractor;
  private readonly threshold: number;
  private readonly llmForNonLatin: boolean;

  constructor(
    llm: LLM,
    options: HybridEntityExtractorOptions = {},
    local: LocalEntityExtractor = new LocalEntityExtractor(),
  ) {
    this.local = local;
    this.llm = new LLMEntityExtractor(llm);
    this.threshold = options.llmFallbackThreshold ?? 1;
    this.llmForNonLatin = options.llmForNonLatin ?? true;
  }

  async extract(text: string): Promise<Entity[]> {
    const localEnts = await this.local.extract(text);
    const needsLlm =
      localEnts.length < this.threshold ||
      (this.llmForNonLatin && NON_LATIN_RE.test(text));
    if (!needsLlm) return localEnts;
    const llmEnts = await this.llm.extract(text);
    if (llmEnts.length === 0) return localEnts;
    return this.merge(localEnts, llmEnts);
  }

  async extractBatch(texts: string[]): Promise<Entity[][]> {
    const out: Entity[][] = [];
    for (const t of texts) out.push(await this.extract(t));
    return out;
  }

  private merge(a: Entity[], b: Entity[]): Entity[] {
    const seen = new Set<string>();
    const out: Entity[] = [];
    for (const e of [...a, ...b]) {
      const key = `${e.type}::${e.text.toLowerCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(e);
    }
    return out;
  }
}
