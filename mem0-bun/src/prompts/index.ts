/**
 * Prompts for fact extraction and memory updates.
 * Mirrors the Python `mem0/configs/prompts.py` core flow.
 */

export function factRetrievalPrompt(today: string = new Date().toISOString().slice(0, 10)): string {
  return `You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.

Types of Information to Remember:
1. Personal Preferences (likes, dislikes, food, products, activities, entertainment).
2. Important Personal Details (names, relationships, dates).
3. Plans and Intentions (events, trips, goals).
4. Activity and Service Preferences (dining, travel, hobbies).
5. Health and Wellness (dietary restrictions, fitness routines).
6. Professional Details (job titles, work habits, career goals).
7. Miscellaneous (favorite books, movies, brands).

Few-shot examples:
Input: Hi.
Output: {"facts": []}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {"facts": ["Looking for a restaurant in San Francisco"]}

Input: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Output: {"facts": ["Had a meeting with John at 3pm", "Discussed the new project"]}

Input: Hi, my name is John. I am a software engineer.
Output: {"facts": ["Name is John", "Is a Software engineer"]}

Return JSON of the form {"facts": ["..."]}.

Rules:
- Today's date is ${today}.
- Do not echo the few-shot examples.
- Extract from user/assistant messages only; ignore system messages.
- If nothing is relevant return {"facts": []}.
- Detect the language of the user input and record facts in the same language.`;
}

export const UPDATE_MEMORY_PROMPT = `You are a smart memory manager controlling a long-term memory store.

You can perform four operations against the existing memory:
- ADD: a new fact not already present.
- UPDATE: an existing fact whose information has changed; keep its original id.
- DELETE: an existing fact contradicted by the new facts.
- NONE: no change (already present or irrelevant).

Compare the retrieved facts with the existing memory list, then return STRICT JSON in this format:
{
  "memory": [
    { "id": "<existing or new>", "text": "<fact>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<only if UPDATE>" }
  ]
}

Guidelines:
- For ADD events, generate a new id (the caller will replace it with a UUID).
- For UPDATE events, return the SAME id from the existing memory.
- For DELETE events, return the SAME id from the existing memory.
- For NONE events, copy the existing entry unchanged.
- Do not invent new facts. Only operate on the inputs provided.
- Output ONLY JSON. No prose.`;

export function buildUpdateMemoryUserPrompt(
  existing: Array<{ id: string; text: string }>,
  retrieved: string[],
  customInstructions?: string,
): string {
  const parts: string[] = [];
  if (customInstructions) {
    parts.push(`Custom instructions:\n${customInstructions}\n`);
  }
  parts.push(`Existing memory:\n${JSON.stringify(existing, null, 2)}\n`);
  parts.push(`Retrieved facts:\n${JSON.stringify(retrieved, null, 2)}\n`);
  parts.push(
    `Return the new memory state as JSON of the form {"memory":[{"id":"...","text":"...","event":"...","old_memory":"..."}]}.`,
  );
  return parts.join("\n");
}
