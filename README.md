# Deepmail

An MCP server that gives AI agents deep understanding of your email -- contacts, relationships, family, company, investments, and more.

## What it does

Deepmail scans your Gmail (read-only) and builds persistent profiles of the people in your life. Once configured, any MCP-compatible AI agent (Claude Code, Claude Desktop, Cursor, etc.) can look up people, understand your relationships, and use your email history as context.

### Available tools

| Tool | Description |
|------|-------------|
| `check_auth` | Verify Gmail authentication status |
| `who_is(person)` | Look up a person from cached profiles |
| `about_me(topic)` | Get context about the user ("overview", "family", "team", etc.) |
| `build_profile(query)` | Start a background Gmail scan (~5 min; returns immediately) |
| `build_status()` | Check progress of a running build |
| `profile_health()` | Check freshness and coverage of cached profiles |
| `get_candidates()` | Return structured candidates from the latest build for review |
| `search_emails(query)` | Lightweight Gmail search (sender/date/subject/snippet) |
| `read_email(message_id)` | Fetch the full body of a specific email by message ID |
| `reset_profile(confirm)` | Wipe all generated data and start fresh |

## Quick start

### 1. Install

```bash
pip install deepmail
```

Or run directly without installing:

```bash
uvx deepmail
```

### 2. Set up Google Cloud credentials

You need a Google Cloud project with the Gmail API enabled. See [Google Cloud Setup](#google-cloud-setup) below for detailed steps.

### 3. Authenticate with Gmail

```bash
deepmail auth
```

This opens a browser for Google OAuth. The token is stored locally and never leaves your machine.

### 4. Configure your AI agent

Run `deepmail init` to write a `.mcp.json` in your project, or add manually:

```json
{
  "mcpServers": {
    "deepmail": {
      "type": "stdio",
      "command": "uvx",
      "args": ["deepmail"]
    }
  }
}
```

Or run `deepmail setup` for a full interactive walkthrough that handles credentials, auth, and agent config.

### 5. Use it

Once configured, your AI agent will automatically:

- Check your email context at session start
- Look up people when you mention them by name
- Search your Gmail when investigating topics
- Build deep profiles of contacts on request

## Google Cloud Setup

Deepmail uses the Gmail API with OAuth 2.0. You need to create a Google Cloud project and obtain OAuth credentials. This is a one-time setup that takes about 5 minutes.

### Step 1: Create a Google Cloud project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click the project dropdown at the top of the page
3. Click **New Project**
4. Enter a name (e.g., "pi-email") and click **Create**
5. Make sure the new project is selected in the dropdown

### Step 2: Enable the Gmail API

1. Go to **APIs & Services > Library** (or search "Gmail API" in the top search bar)
2. Find **Gmail API** and click on it
3. Click **Enable**

### Step 3: Configure the OAuth consent screen

1. Go to **APIs & Services > OAuth consent screen**
2. Select **External** as the user type (unless you have a Google Workspace org) and click **Create**
3. Fill in the required fields:
   - **App name**: deepmail (or anything you like)
   - **User support email**: your email address
   - **Developer contact**: your email address
4. Click **Save and Continue**
5. On the **Scopes** page, click **Add or Remove Scopes**
6. Search for and add `https://www.googleapis.com/auth/gmail.readonly`
7. Click **Update**, then **Save and Continue**
8. On the **Test users** page, click **Add Users** and add your Gmail address
9. Click **Save and Continue**, then **Back to Dashboard**

### Step 4: Create OAuth credentials

1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials > OAuth client ID**
3. Select **Desktop app** as the application type
4. Enter a name (e.g., "deepmail desktop") and click **Create**
5. You will see a dialog with your **Client ID** and **Client Secret** -- copy the Client ID

### Step 5: Configure Deepmail

Set your Client ID as an environment variable:

```bash
export GOOGLE_CLIENT_ID="your-client-id-here.apps.googleusercontent.com"
```

Or create a `.env` file in your working directory:

```
GOOGLE_CLIENT_ID=your-client-id-here.apps.googleusercontent.com
```

Then run authentication:

```bash
deepmail auth
```

Or use the interactive setup wizard which handles everything:

```bash
deepmail setup
```

### Important notes about Google OAuth in testing mode

- **Testing mode**: Your app starts in "Testing" mode on Google Cloud. This is fine for personal use.
- **Token expiry**: In testing mode, OAuth tokens expire every **7 days**. When they expire, re-run `deepmail auth` to re-authenticate.
- **Publishing**: If you want tokens that don't expire weekly, you can submit your app for verification on the OAuth consent screen. This is optional for personal use.
- **Permissions**: Deepmail only requests `gmail.readonly` -- it cannot send, modify, or delete any emails.

## MCP configuration

### Installed from PyPI (recommended)

```json
{
  "mcpServers": {
    "deepmail": {
      "type": "stdio",
      "command": "uvx",
      "args": ["deepmail"]
    }
  }
}
```

Or just run `deepmail init` to write this automatically.

### Local development (from a clone of this repo)

```json
{
  "mcpServers": {
    "deepmail": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/pi-email-deep-context-library",
        "deepmail"
      ]
    }
  }
}
```

Replace `/path/to/pi-email-deep-context-library` with the actual path on your machine.

## Install as a Skill

```bash
npx skills add dennison/pi-email-deep-context-library
```

This installs the `deepmail` skill, which teaches agents the investigation methodology for using the MCP tools. You still need the MCP server configured (see above).

## How it works

1. **OAuth**: Connects to Gmail with read-only access via Google's OAuth 2.0 flow
2. **Search**: Uses Gmail's search API to find relevant emails based on queries
3. **Extract**: Parses email content, strips quotes, extracts entities (names, relationships, organizations)
4. **Embed**: Builds local embeddings for entity canonicalization and deduplication
5. **Profile**: Materializes structured profiles of contacts and relationships into local Markdown files
6. **Serve**: Exposes everything through MCP tools that any compatible AI agent can call

All processing happens locally. Profiles are cached so subsequent lookups are instant.

## Security

- **Read-only**: Deepmail only requests `gmail.readonly` permission. It cannot send, modify, or delete emails.
- **Local-first**: All data (profiles, embeddings, OAuth tokens) stays on your machine. Nothing is uploaded to any server.
- **No telemetry**: Deepmail does not phone home, track usage, or collect any analytics.
- **LLM calls**: If `ANTHROPIC_API_KEY` is set, the build pipeline uses Claude to judge candidate relationships. This is optional -- without it, the calling agent (Claude Code, Cursor, etc.) reviews candidates instead.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) for details.
