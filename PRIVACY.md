# Privacy Policy

**Last updated: May 2026**

## What Deepmail does

Deepmail is an open-source tool that helps AI agents understand your email. It connects to your Gmail account with read-only access to search and analyze your messages locally on your computer.

## What data we access

Deepmail requests `gmail.readonly` access to your Google account. This allows it to:
- Search your Gmail messages
- Read email content (sender, subject, body)

Deepmail **cannot** send, modify, or delete your emails.

## Where your data goes

**Nowhere.** Deepmail runs entirely on your local machine.

- Email content is processed locally and never uploaded to any server
- OAuth tokens are stored locally on your filesystem with restricted permissions
- Cached profiles and search results are stored locally
- No analytics, telemetry, or tracking of any kind

## Third-party services

Deepmail may optionally use the Anthropic API (Claude) for entity classification if you configure an API key. In that case, email excerpts are sent to Anthropic's API for classification. This is optional — the tool works without it, with the calling AI agent performing classification instead.

## Data deletion

Run `deep-email reset_profile --confirm yes` to delete all cached data. Delete your OAuth tokens by removing the files in your platform's application data directory. Revoke Deepmail's access to your Google account at [myaccount.google.com/permissions](https://myaccount.google.com/permissions).

## Open source

Deepmail's source code is publicly available at [github.com/dennisonbertram/deepmail](https://github.com/dennisonbertram/deepmail). You can audit exactly what the tool does with your data.

## Contact

For privacy questions: dennison@withtally.com
