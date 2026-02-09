/**
 * OpenClaw Memory V2 Plugin
 *
 * Phase 1: Structured memory with typed entries, importance scores, tags (keyword search)
 * Phase 2: Local embeddings via @xenova/transformers for semantic search
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";

// ============================================================================
// Types
// ============================================================================

type MemoryType = "learning" | "decision" | "interaction" | "event" | "insight";

type V2MemoryEntry = {
  id: string;
  timestamp: string;
  type: MemoryType;
  importance: number;
  content: string;
  file: string;
  line: number;
  tags: string[];
  context?: string;
  embedding?: number[];
};

type V2MemoryIndex = {
  version: string;
  lastUpdated: string;
  memories: V2MemoryEntry[];
  embeddingModel?: string;
};

type EmbeddingProvider = "none" | "local";

type MemoryV2Config = {
  indexPath: string;
  embedding: {
    provider: EmbeddingProvider;
    modelName?: string;
  };
};

// ============================================================================
// Config
// ============================================================================

function parseConfig(value: unknown): MemoryV2Config {
  const defaultPath = path.join(
    homedir(),
    ".openclaw",
    "workspace",
    "memory",
    "index",
    "memory-index.json",
  );

  const defaults: MemoryV2Config = {
    indexPath: defaultPath,
    embedding: {
      provider: "none",
      modelName: "Xenova/all-MiniLM-L6-v2",
    },
  };

  if (!value || typeof value !== "object") {
    return defaults;
  }

  const cfg = value as Record<string, unknown>;
  const indexPath = typeof cfg.indexPath === "string" ? cfg.indexPath : defaultPath;

  const resolved = indexPath.startsWith("~") ? path.join(homedir(), indexPath.slice(1)) : indexPath;

  const embCfg = cfg.embedding as Record<string, unknown> | undefined;
  const embedding: MemoryV2Config["embedding"] = { ...defaults.embedding };

  if (embCfg) {
    if (embCfg.provider === "local") {
      embedding.provider = embCfg.provider;
    }
    if (typeof embCfg.modelName === "string") {
      embedding.modelName = embCfg.modelName;
    }
  }

  return { indexPath: resolved, embedding };
}

// ============================================================================
// Embeddings (Phase 2) - Native Node.js via @xenova/transformers
// ============================================================================

type Pipeline = (
  text: string,
  options?: { pooling?: string; normalize?: boolean },
) => Promise<{ data: Float32Array }>;

class NodeEmbedder {
  private pipeline: Pipeline | null = null;
  private loading: Promise<void> | null = null;
  private modelName: string;

  constructor(modelName: string) {
    this.modelName = modelName;
  }

  async initialize(): Promise<boolean> {
    if (this.pipeline) return true;
    if (this.loading) {
      await this.loading;
      return !!this.pipeline;
    }

    this.loading = (async () => {
      try {
        // Dynamic import to avoid loading if not needed
        const { pipeline } = await import("@xenova/transformers");
        this.pipeline = (await pipeline("feature-extraction", this.modelName)) as Pipeline;
      } catch (err) {
        console.error("Failed to load embedding model:", err);
        this.pipeline = null;
      }
    })();

    await this.loading;
    return !!this.pipeline;
  }

  async embed(text: string): Promise<number[] | null> {
    if (!this.pipeline) {
      const ok = await this.initialize();
      if (!ok) return null;
    }

    try {
      const output = await this.pipeline!(text, { pooling: "mean", normalize: true });
      return Array.from(output.data);
    } catch (err) {
      console.error("Embedding failed:", err);
      return null;
    }
  }

  isReady(): boolean {
    return !!this.pipeline;
  }
}

// ============================================================================
// Vector Math
// ============================================================================

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;

  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ============================================================================
// Index Operations
// ============================================================================

function loadIndex(indexPath: string): V2MemoryIndex {
  try {
    const content = fs.readFileSync(indexPath, "utf-8");
    return JSON.parse(content) as V2MemoryIndex;
  } catch {
    return { version: "2.0", lastUpdated: new Date().toISOString(), memories: [] };
  }
}

function saveIndex(indexPath: string, index: V2MemoryIndex): void {
  const dir = path.dirname(indexPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(indexPath, JSON.stringify(index, null, 2));
}

// ============================================================================
// Search
// ============================================================================

function scoreKeywordMatch(entry: V2MemoryEntry, queryTerms: string[]): number {
  const content = entry.content.toLowerCase();
  const tags = entry.tags.map((t) => t.toLowerCase());
  const type = entry.type.toLowerCase();

  let score = 0;
  let matchedTerms = 0;

  for (const term of queryTerms) {
    const termLower = term.toLowerCase();
    if (content.includes(termLower)) {
      score += 0.4;
      matchedTerms++;
    }
    if (tags.some((tag) => tag.includes(termLower))) {
      score += 0.3;
      matchedTerms++;
    }
    if (type.includes(termLower)) {
      score += 0.2;
      matchedTerms++;
    }
  }

  if (queryTerms.length > 0) {
    score = score / queryTerms.length;
  }

  score += (entry.importance / 10) * 0.2;

  if (matchedTerms >= queryTerms.length) {
    score += 0.1;
  }

  return Math.min(score, 1.0);
}

function searchMemories(
  indexPath: string,
  query: string,
  maxResults: number,
  minScore: number,
  queryEmbedding?: number[] | null,
): Array<{ entry: V2MemoryEntry; score: number; keywordScore: number; semanticScore: number }> {
  const index = loadIndex(indexPath);
  const queryTerms = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1);

  if (queryTerms.length === 0 && !queryEmbedding) {
    return [];
  }

  const scored: Array<{
    entry: V2MemoryEntry;
    score: number;
    keywordScore: number;
    semanticScore: number;
  }> = [];

  for (const entry of index.memories) {
    const keywordScore = queryTerms.length > 0 ? scoreKeywordMatch(entry, queryTerms) : 0;

    let semanticScore = 0;
    if (queryEmbedding && entry.embedding) {
      semanticScore = cosineSimilarity(queryEmbedding, entry.embedding);
    }

    let score: number;
    if (queryEmbedding && entry.embedding) {
      // Hybrid: 60% semantic, 40% keyword
      score = semanticScore * 0.6 + keywordScore * 0.4;
    } else {
      score = keywordScore;
    }

    score += (entry.importance / 10) * 0.1;

    if (score >= minScore) {
      scored.push({ entry, score, keywordScore, semanticScore });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, maxResults);
}

// ============================================================================
// Plugin
// ============================================================================

const memoryV2Plugin = {
  id: "memory-v2",
  name: "Memory V2",
  description: "Structured memory with typed entries, importance scores, tags, and semantic search",
  kind: "memory" as const,

  configSchema: {
    type: "object" as const,
    additionalProperties: false,
    properties: {
      indexPath: { type: "string" },
      embedding: {
        type: "object",
        additionalProperties: false,
        properties: {
          provider: { type: "string", enum: ["local", "none"] },
          modelName: { type: "string" },
        },
      },
    },
  },

  register(api: OpenClawPluginApi) {
    const config = parseConfig(api.pluginConfig);

    // Initialize embedder if configured
    let embedder: NodeEmbedder | null = null;
    if (config.embedding.provider === "local") {
      embedder = new NodeEmbedder(config.embedding.modelName || "Xenova/all-MiniLM-L6-v2");
      // Pre-warm the model in background
      embedder.initialize().then((ok) => {
        if (ok) api.logger.info("memory-v2: embedding model loaded");
        else
          api.logger.warn(
            "memory-v2: embedding model failed to load, falling back to keyword search",
          );
      });
    }

    api.logger.info(`memory-v2: plugin loaded (embeddings: ${config.embedding.provider})`);

    // ========================================================================
    // Tool: memory_v2_search
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_search",
        label: "Memory Search (V2)",
        description:
          "Search structured memories. Uses hybrid keyword + semantic search when embeddings are enabled.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          maxResults: Type.Optional(Type.Number({ description: "Max results (default: 10)" })),
          minScore: Type.Optional(
            Type.Number({ description: "Min score threshold (default: 0.1)" }),
          ),
        }),
        async execute(_toolCallId, params) {
          const query = typeof params.query === "string" ? params.query : "";
          const maxResults = typeof params.maxResults === "number" ? params.maxResults : 10;
          const minScore = typeof params.minScore === "number" ? params.minScore : 0.1;

          // Get query embedding if available
          let queryEmbedding: number[] | null = null;
          if (embedder) {
            queryEmbedding = await embedder.embed(query);
          }

          const results = searchMemories(
            config.indexPath,
            query,
            maxResults,
            minScore,
            queryEmbedding,
          );

          const searchMode = queryEmbedding ? "hybrid" : "keyword";
          const formatted = results.map(({ entry, score, keywordScore, semanticScore }) => ({
            id: entry.id,
            path: entry.file.startsWith("daily/") ? `memory/${entry.file}` : entry.file,
            line: entry.line,
            score: Math.round(score * 100) / 100,
            keywordScore: Math.round(keywordScore * 100) / 100,
            semanticScore: Math.round(semanticScore * 100) / 100,
            type: entry.type,
            importance: entry.importance,
            tags: entry.tags,
            snippet: entry.content,
            hasEmbedding: !!entry.embedding,
          }));

          return {
            content: [
              {
                type: "text",
                text:
                  results.length > 0
                    ? `Found ${results.length} memories (${searchMode}):\n\n${results
                        .map(
                          (r, i) =>
                            `${i + 1}. [${r.entry.type.toUpperCase()}] (importance: ${r.entry.importance}, score: ${(r.score * 100).toFixed(0)}%) ${r.entry.content.slice(0, 100)}...`,
                        )
                        .join("\n")}`
                    : `No matching memories found (${searchMode}).`,
              },
            ],
            details: { results: formatted, searchMode },
          };
        },
      },
      { name: "memory_v2_search" },
    );

    // ========================================================================
    // Tool: memory_v2_get
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_get",
        label: "Memory Get (V2)",
        description: "Read content from a memory file.",
        parameters: Type.Object({
          path: Type.String({ description: "Path to memory file" }),
          from: Type.Optional(Type.Number({ description: "Starting line number" })),
          lines: Type.Optional(Type.Number({ description: "Number of lines to read" })),
        }),
        async execute(_toolCallId, params) {
          const relPath = typeof params.path === "string" ? params.path : "";
          const from = typeof params.from === "number" ? params.from : undefined;
          const numLines = typeof params.lines === "number" ? params.lines : undefined;

          let absPath = relPath;
          if (relPath.startsWith("~")) {
            absPath = path.join(homedir(), relPath.slice(1));
          } else if (!path.isAbsolute(relPath)) {
            absPath = path.join(homedir(), ".openclaw", "workspace", relPath);
          }

          try {
            const content = fs.readFileSync(absPath, "utf-8");
            const allLines = content.split("\n");

            let text: string;
            if (from !== undefined || numLines !== undefined) {
              const startLine = (from ?? 1) - 1;
              const endLine = numLines ? startLine + numLines : allLines.length;
              text = allLines.slice(startLine, endLine).join("\n");
            } else {
              text = content;
            }

            return {
              content: [{ type: "text", text }],
              details: { path: relPath },
            };
          } catch {
            return {
              content: [{ type: "text", text: `Could not read file: ${relPath}` }],
              details: { path: relPath, error: true },
            };
          }
        },
      },
      { name: "memory_v2_get" },
    );

    // ========================================================================
    // Tool: memory_v2_embed
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_embed",
        label: "Memory Embed (V2)",
        description: "Generate embeddings for memories. Run this to enable semantic search.",
        parameters: Type.Object({
          limit: Type.Optional(
            Type.Number({ description: "Max memories to embed (default: 100)" }),
          ),
          force: Type.Optional(Type.Boolean({ description: "Re-embed even if embedding exists" })),
        }),
        async execute(_toolCallId, params) {
          if (!embedder) {
            return {
              content: [
                {
                  type: "text",
                  text: "Embeddings not configured. Set embedding.provider to 'local' in plugin config.",
                },
              ],
              details: { error: true },
            };
          }

          const ready = await embedder.initialize();
          if (!ready) {
            return {
              content: [
                {
                  type: "text",
                  text: "Failed to load embedding model. Check that @xenova/transformers is installed.",
                },
              ],
              details: { error: true },
            };
          }

          const limit = typeof params.limit === "number" ? params.limit : 100;
          const force = params.force === true;

          const index = loadIndex(config.indexPath);
          let embedded = 0,
            skipped = 0,
            failed = 0;

          for (const memory of index.memories) {
            if (embedded >= limit) break;

            if (memory.embedding && !force) {
              skipped++;
              continue;
            }

            const embedding = await embedder.embed(memory.content);
            if (embedding) {
              memory.embedding = embedding;
              embedded++;
            } else {
              failed++;
            }
          }

          if (embedded > 0) {
            index.embeddingModel = config.embedding.modelName;
            saveIndex(config.indexPath, index);
          }

          return {
            content: [
              {
                type: "text",
                text: `Embedding complete: ${embedded} embedded, ${skipped} already had embeddings, ${failed} failed.\nSemantic search is now ${embedded > 0 || skipped > 0 ? "enabled" : "disabled"}.`,
              },
            ],
            details: { embedded, skipped, failed, model: config.embedding.modelName },
          };
        },
      },
      { name: "memory_v2_embed" },
    );

    // ========================================================================
    // Tool: memory_v2_stats
    // ========================================================================
    api.registerTool(
      {
        name: "memory_v2_stats",
        label: "Memory Stats (V2)",
        description: "Show memory index statistics.",
        parameters: Type.Object({}),
        async execute() {
          const index = loadIndex(config.indexPath);

          const byType: Record<string, number> = {};
          const byImportance: Record<string, number> = {};
          let withEmbeddings = 0;

          for (const m of index.memories) {
            byType[m.type] = (byType[m.type] || 0) + 1;
            const bucket =
              m.importance >= 8 ? "high (8-10)" : m.importance >= 5 ? "medium (5-7)" : "low (1-4)";
            byImportance[bucket] = (byImportance[bucket] || 0) + 1;
            if (m.embedding) withEmbeddings++;
          }

          const pct =
            index.memories.length > 0
              ? Math.round((withEmbeddings / index.memories.length) * 100)
              : 0;

          return {
            content: [
              {
                type: "text",
                text: `Memory V2 Stats:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total memories: ${index.memories.length}
With embeddings: ${withEmbeddings}/${index.memories.length} (${pct}%)
Embedding model: ${index.embeddingModel || "none"}
Provider: ${config.embedding.provider}

By type: ${Object.entries(byType)
                  .map(([k, v]) => `${k}:${v}`)
                  .join(", ")}
By importance: ${Object.entries(byImportance)
                  .map(([k, v]) => `${k}:${v}`)
                  .join(", ")}

${pct < 100 && config.embedding.provider === "local" ? "💡 Run memory_v2_embed to enable semantic search for all memories." : ""}`,
              },
            ],
            details: {
              total: index.memories.length,
              withEmbeddings,
              embeddingModel: index.embeddingModel,
              provider: config.embedding.provider,
              byType,
              byImportance,
            },
          };
        },
      },
      { name: "memory_v2_stats" },
    );
  },
};

export default memoryV2Plugin;
