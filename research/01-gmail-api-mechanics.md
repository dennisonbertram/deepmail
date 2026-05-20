# Gmail API Mechanics for a Search-and-Bulk-Download Pipeline

*Research date: 2026-05-16. All quota numbers below reflect the **post-May 2026** schedule, which applies to any new GCP project; older projects keep the prior limits until April 2026 grandfather expires.*

## TL;DR

- **Deterministic bulk download is feasible** for the design described, using the Gmail REST API: paginate `users.messages.list` with a Gmail-search `q=`, then resolve every ID with batched `users.messages.get`. There is **no documented total-results cap** on pagination — `resultSizeEstimate` is only an estimate, so trust the `nextPageToken` cursor.
- **Throughput ceiling per OAuth user is ~300 message-fetches per minute** under the new quota (6,000 quota units/min/user ÷ 20 units per `messages.get`). That is **~18,000 messages/hour**, or ~430k/day before you trip the per-user-per-minute limit. Project-wide daily cap is 80M units (~4M messages). Plenty for a single mailbox; a non-issue at hobby scale.
- **The expensive call is `messages.get`, not `messages.list`.** Listing is cheap (5 units, 500 IDs/page); fetching content costs 20 units (4× more than the 2024 docs). Batching reduces HTTP round-trip cost but **does not reduce quota cost** — each sub-request still bills individually.
- **`format=full` is the right default** for an LLM-grep-markdown pipeline. `raw` is full RFC 2822 base64url, which means re-parsing MIME yourself for nothing. `metadata` is forbidden under `gmail.readonly` is fine, but `gmail.metadata` blocks body access entirely. `minimal` is only useful for cache-revalidation.
- **The biggest deal-breakers are auth-related, not API-mechanic-related:** `gmail.readonly` is a *restricted* scope, so a production app needs OAuth verification + a third-party security assessment. For solo use, "Testing" mode works but the refresh token expires every 7 days.

## API surface choice: REST vs IMAP vs Takeout

The REST API is the only realistic option for this workflow.

| | REST (`gmail.googleapis.com/v1`) | IMAP | Google Takeout |
|---|---|---|---|
| Search-server-side w/ Gmail operators | Yes (`q=`) | Limited (IMAP SEARCH ≠ Gmail operators) | No |
| Incremental | Yes (`history.list`) | Polling-only | No, full re-export |
| Quota model | Quota units (predictable) | Per-user bandwidth + concurrent-connection caps, less documented | One-shot, hours-to-deliver |
| Verification burden for production | Restricted-scope review | Same (uses XOAUTH2) | None (manual download) |
| Body format | JSON-parsed `payload` or `raw` MIME | RFC 2822 | mbox in a zip |

IMAP is a non-starter because Gmail's search syntax (`from:`, `has:attachment`, `label:`, `older_than:`) is what makes the deterministic-query thesis work — IMAP SEARCH doesn't have equivalents for label/`has:` filters. Takeout is fine for a one-time corpus snapshot but not for an iterative grep-then-search-again loop.

Source: [Gmail API guides — list-messages](https://developers.google.com/workspace/gmail/api/guides/list-messages), [filtering](https://developers.google.com/workspace/gmail/api/guides/filtering).

## Quotas (the load-bearing section)

From [developers.google.com/workspace/gmail/api/reference/quota](https://developers.google.com/workspace/gmail/api/reference/quota):

| Limit | Value | Error |
|---|---|---|
| Per-project per-minute | 1,200,000 units | `rateLimitExceeded` (403) |
| **Per-user per-minute** | **6,000 units** *(was 15,000 pre-2026)* | `userRateLimitExceeded` (403) |
| **Per-project per-day** | **80,000,000 units** *(new, uncappable)* | `dailyLimitExceeded` (403) |

Per-method costs that matter here:

| Method | Units | Notes |
|---|---|---|
| `messages.list` | 5 | Up to 500 IDs/page |
| `messages.get` | 20 | *Was 5 pre-2026.* This is the dominant cost. |
| `messages.attachments.get` | 20 | Avoid unless we need them. |
| `threads.get` | 40 | Returns all messages in thread. Beats `messages.get` for threads of 3+. |
| `history.list` | 2 | For incremental sync. |

**Practical ceiling per user**: 6,000 / 20 = 300 `messages.get` calls/min = **18,000/hour, ~432,000/day** before backoff. The 80M project-day cap means you could theoretically download ~4M messages/day if you had multiple users.

**Batching does not save quota** — Google bills each sub-request inside a batch separately. Batch is purely for HTTP overhead. Sweet spot is 50 sub-requests; the hard cap is 100, and batches >50 are documented to "trigger rate limiting." Source: [batch guide](https://developers.google.com/workspace/gmail/api/guides/batch), [handle-errors](https://developers.google.com/workspace/gmail/api/guides/handle-errors).

Retry strategy: exponential backoff starting at ≥1s for 403 `userRateLimitExceeded`, 429, and 5xx.

## Pagination

`users.messages.list` returns up to `maxResults` IDs/page (default 100, max **500**) plus a `nextPageToken`. No documented total-result cap — keep pulling until `nextPageToken` is absent. `resultSizeEstimate` is explicitly an *estimate* and is widely reported to be wrong by 10–30% in practice; only trust the page iteration. Source: [users.messages.list reference](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/list).

## Message payload format

Per the [Format enum](https://developers.google.com/workspace/gmail/api/reference/rest/v1/Format):

- `minimal` — ID + labels only. Use for cache revalidation.
- `metadata` — ID + labels + headers, no body. Cheap if you just need who/when/subject.
- **`full`** — parsed `payload` tree (headers + MIME parts with body.data base64url-encoded inline for small parts, `attachmentId` pointer for large parts). **Best default for LLM-readable markdown conversion** — no MIME parser needed.
- `raw` — full RFC 2822 as one base64url string. Use only if you want to feed it to `email.parser.BytesParser` yourself or re-emit via SMTP.

`full` and `raw` are blocked under the `gmail.metadata` scope. All formats cost the same 20 units.

For markdown conversion: walk `payload.parts[]`, prefer `mimeType=text/plain`, fall back to `text/html` with html-to-md, skip parts whose `body.attachmentId` is set unless you want them.

## Threads

`users.threads.get?format=full` returns every message in the thread in chronological order with full payloads — clean reply structure, no need to reconstruct from `References` / `In-Reply-To`. Costs 40 units regardless of thread size, so it beats `N × messages.get` (20 each) for any thread with 3+ messages. For a search-and-download pipeline, fetching by thread is usually the win.

Caveat: the search `q=` parameter operates on messages, not threads (the API does **not** do thread-wide matching the way the Gmail UI does). You can list threads instead via `users.threads.list?q=` — that does match any-message-in-thread.

## Attachments

Inline in `payload.parts[].body.data` if small (~few KB); otherwise only `attachmentId` is present and you need a separate `users.messages.attachments.get` call (20 units, returns base64url). For this project, **don't fetch them** — markdown grep-over-text doesn't benefit from attachment bytes. Just keep filename + MIME type as metadata.

## Auth

Use `https://www.googleapis.com/auth/gmail.readonly`. Pitfalls per [OAuth scopes page](https://developers.google.com/workspace/gmail/api/auth/scopes) and [OAuth expiration docs](https://developers.google.com/identity/protocols/oauth2):

- `gmail.readonly` is a **restricted** scope → requires OAuth verification + a CASA security assessment (~$10–75k for an external auditor) **if you put it in production**. For solo/self-use it can stay "Testing".
- "Testing" publishing status issues a refresh token that **expires every 7 days**. You'll re-consent weekly. The 7-day rule does not apply to `userinfo.email`/`profile`/`openid` only — but it does apply to any Gmail scope.
- 100 refresh tokens per (Google account × OAuth client). Oldest gets silently invalidated past 100.
- Installed-app flow (loopback redirect, `run_local_server`) is the path of least resistance for a personal tool.

## Search-operator gotchas

Per [filtering guide](https://developers.google.com/workspace/gmail/api/guides/filtering):

- All `after:` / `before:` dates are **interpreted as midnight PST**, not UTC and not the user's TZ. Pass Unix seconds (`after:1704067200`) to pin a timezone.
- API does **not** expand Workspace aliases. `from:primary@x.com` will miss messages sent from `alias@x.com` that the UI would catch.
- API does **not** do thread-wide search on `messages.list` (use `threads.list` for that behavior).
- Otherwise all UI operators (`from:`, `to:`, `subject:`, `has:attachment`, `label:`, `is:`, `in:`, `larger:`, `older_than:`, `filename:`, etc.) work.

## POC sketch

```python
from googleapiclient.discovery import build
from googleapiclient.http import BatchHttpRequest
from google.oauth2.credentials import Credentials
import base64, json

svc = build("gmail", "v1", credentials=Credentials.from_authorized_user_file("token.json"))

def fetch_all_matching(query: str, out_path: str):
    ids, page = [], None
    while True:
        resp = svc.users().messages().list(
            userId="me", q=query, maxResults=500, pageToken=page
        ).execute()
        ids.extend(m["id"] for m in resp.get("messages", []))
        page = resp.get("nextPageToken")
        if not page:
            break

    results = {}
    def cb(req_id, response, exception):
        if exception is None:
            results[req_id] = response

    # 50 per batch; each get still bills 20 quota units
    for chunk_start in range(0, len(ids), 50):
        batch = svc.new_batch_http_request(callback=cb)
        for mid in ids[chunk_start:chunk_start + 50]:
            batch.add(
                svc.users().messages().get(userId="me", id=mid, format="full"),
                request_id=mid,
            )
        batch.execute()  # add exponential backoff on HttpError 403/429/5xx in real code

    with open(out_path, "w") as f:
        for mid, msg in results.items():
            f.write(json.dumps(msg) + "\n")
```

## Open questions to test before committing

1. **Real-world `resultSizeEstimate` drift** — page through a large query and compare estimate vs actual count.
2. **Quota burn under 50-batch parallelism** — does 50 concurrent gets really stay under 6,000/min, or does Google's "concurrent request" 429 kick in earlier?
3. **`threads.list?q=` vs `messages.list?q=` semantics** — does threads.list return one thread even when multiple messages in it match? (Saves duplicate work.)
4. **Encoding/decoding edge cases** — what does `body.data` look like for non-UTF-8 payloads, S/MIME signed, PGP-encrypted, or text/calendar parts?
5. **7-day token renewal UX** — can we tolerate weekly re-consent in test mode, or do we need verification?
6. **Cost of the new (2026) quotas on this pipeline if the user has >50k messages matching a typical query** — at 300/min that's 3 hours of wall time per pass; need to decide if persistent local cache makes that a one-time cost.
7. **`history.list` for incremental refresh** — confirm we can avoid re-downloading after the initial snapshot, and that `historyId` is stable across our cache TTL.
