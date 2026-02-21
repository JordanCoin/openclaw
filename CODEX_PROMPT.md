# Task: Code Review — How OpenClaw Handles Compaction

Review the OpenClaw codebase to understand how context compaction works. We want to know:

1. **Where does compaction happen?** Find the code that decides when to compact and executes it.
2. **Is there any plugin hook/event before or after compaction?** (e.g., `pre-compact`, `post-compact`, `onCompact`)
3. **How is the compaction summary generated?** What model call happens, what prompt is used?
4. **Is there a way to detect context depth/usage programmatically?** (e.g., token count, percentage)
5. **Are there any extension points where a plugin could intercept the compaction lifecycle?**

Focus on:

- `src/` directory — look for files mentioning "compact", "compaction", "context", "summary"
- `src/gateway/` — session management
- `src/plugins/` or `src/extensions/` — plugin system hooks
- Any event emitter or hook registration patterns

Output a concise report with:

- File paths and line numbers for key compaction code
- Whether pre/post compaction hooks exist
- Whether a plugin can currently intercept compaction
- Suggestions for where hooks COULD be added if they don't exist

Keep it focused — we don't need a full codebase review, just the compaction lifecycle.
