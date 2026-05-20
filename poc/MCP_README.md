# pi-email MCP Server

Gives AI agents (Claude Code, Claude Desktop, Cursor) deep understanding
of your email -- who your family is, your key contacts, your relationships.

## Setup

1. Install dependencies:
   ```bash
   cd /Users/dennison/develop/pi-email-deep-context-library/poc
   uv sync
   ```

2. Authenticate to Gmail (one-time):
   ```bash
   uv run pi-email auth
   ```

3. Add to your project's `.claude/settings.json`:
   ```json
   {
     "mcpServers": {
       "pi-email": {
         "command": "uv",
         "args": ["run", "--directory", "/Users/dennison/develop/pi-email-deep-context-library/poc", "python", "-m", "pi_email.mcp_entry"]
       }
     }
   }
   ```

4. Restart Claude Code. The tools are now available.

## Tools

| Tool | Description |
|------|-------------|
| `check_auth()` | Verify Gmail OAuth tokens exist and are valid |
| `who_is(person)` | Look up a person in the materialized family profile |
| `build_profile(query)` | Run the full expansion pipeline against Gmail (takes several minutes) |
| `build_status()` | Check progress of a running build |
| `about_me(topic)` | Get context about the user ("overview", "family", "team", etc.) |
| `profile_health()` | Check freshness and coverage of cached profiles |
| `get_candidates()` | Return structured candidates for the calling model to review |
| `reset_profile(confirm)` | Wipe all generated data and start fresh; pass `confirm="yes"` to proceed |

## How it works

By default, the MCP server **skips the internal LLM judge** and returns
structured candidates for your AI agent to classify. This means:

- No `ANTHROPIC_API_KEY` needed in the server environment
- No extra API cost -- the calling model (which is already an LLM) does the judgment
- The profile contains three sections: auto-accepted, candidates for review, auto-rejected

### After a build completes

1. Call `get_candidates()` to get the structured candidate list
2. Auto-accepted members are already confirmed (surname matches) -- acknowledge them
3. For each "candidates for review" entry:
   - If obviously not family (newsletter mention, public figure): reject silently
   - If probably family (personal context, possessive reference): accept
   - If ambiguous: ask the user to confirm
4. Report the final family list to the user

## Optional: internal LLM judge

If you provide `ANTHROPIC_API_KEY`, the server uses its own LLM judge for
classification. If omitted (recommended), the server returns candidates
for your AI agent to classify -- no extra API cost.

```json
{
  "mcpServers": {
    "pi-email": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/dennison/develop/pi-email-deep-context-library/poc", "python", "-m", "pi_email.mcp_entry"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

## Usage

In Claude Code, try:
- "Check if pi-email is authenticated" -> calls `check_auth`
- "Who is Jana Bertram?" -> calls `who_is`
- "Build a profile of my family from my email" -> calls `build_profile` (takes a few minutes)
- "Show me the candidates from the latest build" -> calls `get_candidates`

## Security

- This server reads your Gmail via OAuth (gmail.readonly scope)
- Profile data is stored locally only (never uploaded)
- The `who_is` tool returns only matching sections, never the full profile
- Install in project-level settings only -- do NOT install globally
- When `ANTHROPIC_API_KEY` is provided, entity classification calls go to the Anthropic API
