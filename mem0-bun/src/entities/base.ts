import type { Entity } from "../utils/entity_extraction";

export interface EntityExtractor {
  extract(text: string): Promise<Entity[]>;
  extractBatch(texts: string[]): Promise<Entity[][]>;
}
