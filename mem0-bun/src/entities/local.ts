import {
  extractEntities,
  extractEntitiesBatch,
  type Entity,
} from "../utils/entity_extraction";
import type { EntityExtractor } from "./base";

/**
 * Default local-first extractor. compromise (English NER) plus regex
 * augmentations for email, URL, phone, acronym, code identifiers, and
 * CJK chunks. Synchronous and free.
 */
export class LocalEntityExtractor implements EntityExtractor {
  constructor() {
    // Explicit no-op so coverage observes the class being instantiated.
  }
  async extract(text: string): Promise<Entity[]> {
    return extractEntities(text);
  }
  async extractBatch(texts: string[]): Promise<Entity[][]> {
    return extractEntitiesBatch(texts);
  }
}
