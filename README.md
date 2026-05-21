# Deepmail

**Your agent can search your email. It can't understand it.**

Gmail MCP servers let agents read individual messages. But understanding email — knowing who matters, what relationships exist, what's happening in your life and business — requires investigation, cross-referencing, and memory. Agents can't do that in a single search.

Deepmail gives your AI agent 10 tools to investigate, analyze, and remember your email. It builds persistent knowledge that survives across sessions — so your agent actually knows you.

## What your agent can do with Deepmail

```
You: "Who are my investors?"

Agent searches your Gmail for term sheets, fund communications, board emails.
Agent reads specific emails to understand deal details.
Agent cross-references names across rounds.
Agent builds a complete investor table — firms, key contacts, amounts, dates.

Next session, your agent already knows. No re-searching.
```

## Quick start

```bash
pip install deep-email
deep-email auth
```

Opens your browser for Google consent. Read-only access. Done.

> You may see "This app isn't verified" — click **Advanced** → **Go to Deepmail (unsafe)**.
> This is standard for open-source tools. Deepmail only requests read-only email access.

### Configure your agent

Add to your project's `.mcp.json`:
```json
{
  "mcpServers": {
    "deepmail": {
      "type": "stdio",
      "command": "uvx",
      "args": ["deep-email"]
    }
  }
}
```

Or use the CLI:
```bash
deep-email init   # writes .mcp.json for you
```

### Install the skill (teaches your agent how to use the tools)

```bash
npx skills add dennisonbertram/deepmail
```

Works with Claude Code, Cursor, Codex, GitHub Copilot, Windsurf, Gemini CLI, and 40+ other agents.

<details>
<summary><strong>Advanced: Use your own Google Cloud credentials</strong></summary>

By default, Deepmail uses built-in OAuth credentials. If you prefer to use your own GCP project:

### 1. Create a Google Cloud Project
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- **Select a project** → **New Project** → name it anything → **Create**

### 2. Enable the Gmail API
- Go to [Gmail API](https://console.cloud.google.com/apis/library/gmail.googleapis.com)
- Click **Enable**

### 3. Configure OAuth Consent Screen
- Go to [OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent)
- Choose **External** → **Create**
- App name: anything. Support email: yours. Developer contact: yours.
- **Scopes** → **Add or Remove Scopes** → find `gmail.readonly` → check it → **Update** → **Save**
- **Test users** → **Add Users** → add your email → **Save**

> **Note:** Your app starts in "Testing" mode. Tokens expire every 7 days — re-run `deep-email auth` when they do. For non-expiring tokens, submit your app for verification on the consent screen (optional for personal use).

### 4. Create Credentials
- Go to [Credentials](https://console.cloud.google.com/apis/credentials)
- **Create Credentials** → **OAuth client ID** → **Desktop app** → **Create**
- Copy the **Client ID** and **Client Secret**

### 5. Set the credentials
```bash
export GOOGLE_CLIENT_ID="your-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-secret"
```

Or add to `~/.zshrc` / `~/.bashrc` for persistence. Then run `deep-email auth`.

</details>

## Tools

| Tool | What it does |
|---|---|
| `search_emails(query)` | Search Gmail — returns sender, date, subject, snippet |
| `read_email(message_id)` | Read the full body of a specific email |
| `build_profile(query)` | Deep investigation — extracts people, builds profiles |
| `build_status()` | Check progress of a running investigation |
| `get_candidates()` | Review extracted candidates from latest investigation |
| `who_is(person)` | Look up a person from cached knowledge |
| `about_me(topic)` | What the agent knows about you, by topic |
| `profile_health()` | Check how fresh the cached knowledge is |
| `reset_profile(confirm)` | Wipe cached knowledge and start fresh |
| `check_auth()` | Verify Gmail connection |

## How it works

1. **Your agent reads the skill** and learns how to investigate email topics
2. **It searches your Gmail** using `search_emails` — following leads, reading full emails when needed
3. **It builds profiles** using `build_profile` — extracting people, relationships, and context
4. **Knowledge persists** — `who_is` and `about_me` return cached results instantly in future sessions
5. **It gets smarter** — each investigation adds to the persistent knowledge base

All data stays on your machine. The MCP server runs locally. No cloud storage, no telemetry.

## Examples

**"Figure out my investors"** — Agent searches for term sheets, fund emails, board communications. Builds a table of investors across rounds with key contacts and amounts.

**"Who's on my team?"** — Agent searches internal comms, standup emails, shared docs. Maps your org with roles and last-contact dates.

**"Tell me about my relationship with [person]"** — Agent searches all correspondence, synthesizes a relationship summary with timeline.

**"What should I know before my meeting with [person]?"** — Agent pulls recent threads, open items, relationship context.

## License

AGPL-3.0 — free to use, modify, and distribute. If you run it as a hosted service, you must release your source under the same license. See [LICENSE](./LICENSE).

## Security

- **Read-only**: Only `gmail.readonly` scope. Never sends, modifies, or deletes.
- **Local-first**: All data on your machine. No cloud, no telemetry.
- **OAuth scoped**: Minimum required permissions.
- **Wipe anytime**: `reset_profile(confirm="yes")` deletes all cached data.
