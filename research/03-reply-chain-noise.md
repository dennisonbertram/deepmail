# Reply-chain noise: stripping quotes, signatures, and forwarding cruft from a Gmail corpus

## Recommendation

For a grep + LLM corpus built from a personal Gmail archive, do **per-thread documents with messages stored newest-first and quoted blocks stripped at write time**, using a hybrid stripper: trust Gmail's own thread linkage and `internalDate` ordering as the primary source of truth (you already know what each individual message contained, so quoted lower-down content is structurally redundant), then run a heuristic stripper — [`mail-parser-reply`](https://github.com/alfonsrv/mail-parser-reply) for plain text and a small custom HTML pass that drops `div.gmail_quote`, `blockquote` chains, and `div.OutlookMessageHeader` — over each message before concatenation. This pairs the cleanest signal (thread structure) with a maintained, multilingual heuristic layer, and it sidesteps talon's two real problems (the ML signature model is from 2016 and the project hasn't shipped in nearly a decade). Keep the original raw `.eml` (or the raw markdown) alongside the cleaned corpus so a future re-strip is a one-pass job, not a re-download.

## The problem in concrete form

A 10-message Gmail thread between Alice and Bob, fetched naively, contains roughly `n*(n+1)/2 = 55` copies of the earliest message's content — once in its own message, then quoted in every subsequent reply. Grep over the corpus then counts entities by quoting depth, not relevance: a person mentioned once in a kicked-off long thread outranks a person mentioned in five short threads. Token cost for LLM passes scales the same way.

The noise comes from four interacting conventions:

**Plaintext quoting.** RFC 3676 `>` prefix lines, optionally with an attribution header on the preceding line:
- English Gmail: `On Mon, Jan 15, 2024 at 09:00, Jane <jane@example.com> wrote:`
- French: `Le 15 janv. 2024 à 09:00, Jane <jane@example.com> a écrit :`
- German: `Am 15.01.2024 um 09:00 schrieb Jane <jane@example.com>:`
- Japanese: `2024年1月15日(月) 9:00 Jane <jane@example.com>:`

**HTML quoting.** Gmail wraps quoted content in `<div class="gmail_quote">` containing a `<blockquote class="gmail_quote">` (note: same class on two elements). Outlook uses `<div id="appendonsend">` plus `<div id="divRplyFwdMsg">` and `<hr id="stopSpelling">`. Apple Mail wraps in `<blockquote type="cite">`. Nested replies stack `blockquote` elements, which is the only reliable structural signal across clients.

**Forward envelopes.** `---------- Forwarded message ----------` (Gmail), `Begin forwarded message:` (Apple Mail), `From: ... Sent: ... To: ... Subject:` block (Outlook). Forwards are not redundant the way reply quotes are — the forwarded body may be the only place the content lives — so a stripper must distinguish forward-from-reply, not just delete everything below a delimiter.

**Signatures.** RFC 3676 says `-- ` (dash-dash-space-newline) is the canonical signature delimiter, and almost no real client honors it consistently. Real signatures show up as:
- Mobile auto-appends: `Sent from my iPhone`, `Von meinem iPhone gesendet`, `Get Outlook for Android`
- Plain name/title blocks with no delimiter at all
- HTML signatures with embedded image logos, vCards, and disclaimers
- Corporate confidentiality disclaimers ("This email is confidential...") that are not signatures but live in the same trailing position

## Library landscape

| Library | Language | Approach | Last release | Reputation / notes |
|---|---|---|---|---|
| [talon](https://github.com/mailgun/talon) | Python | Regex for quotes + SVM/ML for signatures; handles HTML | **v1.2.5, Apr 2016** | Historically the gold standard; **effectively unmaintained**, 52 open issues, pre-trained model is ancient. Still works, still imported widely. |
| [email-reply-parser](https://github.com/crisp-oss/email-reply-parser) (Crisp, JS) | JavaScript | Heuristic line/fragment scanner | Actively maintained | ~10 locales (EN, FR, ES, PT, IT, JA, ZH, ...); Crisp claims ~1M inbound emails/day. Plain text only. |
| [email_reply_parser](https://github.com/github/email_reply_parser) (Ruby) | Ruby | Regex line scanner; looks for "on/wrote" + leading `-`/`_` | v0.5.11, May 2023 | English-only by design (README admits this); breaks on Gmail's 80-col-wrapped headers. |
| [email-reply-parser](https://github.com/zapier/email-reply-parser) (Zapier, Python port) | Python | Port of the Ruby one | Sparse | Same English-only heuristics; small but functional. |
| [mail-parser-reply](https://github.com/alfonsrv/mail-parser-reply) | Python | Configurable regex per language; splits into distinct replies | Recent commits, no formal releases | 13 languages (9 tested, 4 untested). Text-only. **Best maintained Python option for multilingual plaintext today.** |
| [quotequail](https://pypi.org/project/quotequail/) | Python | Regex/heuristics; returns structured (`reply`/`forward`/`quote`) with parsed headers | **v0.4.0, Jul 2024** | Both plain text and HTML; from Close.io; small surface area, well-suited for splitting forwards from replies. |

There is no widely-adopted modern ML reply-detector — most projects either still use talon's stale model or settle for regex. A from-scratch fine-tune is not justified for this use case.

## Strategy comparison

| Strategy | Grep precision | LLM token cost | Provenance | Robustness | Simplicity |
|---|---|---|---|---|---|
| **Per-thread, quotes stripped** (recommended) | High — each entity counted once per thread | Lowest | Per-thread (lose per-message byte offsets) | High — relies on Gmail thread linkage, not regex | Medium |
| Per-message + quote stripping | Medium — stripper misses leak | Low | Per-message | Medium — failure modes (inline replies) bite hard | Medium |
| Per-message + delta byte ranges | High if delta is correct | Low at query time | Full | Medium — same stripper-quality dependency | Low (complex offsets) |
| Hybrid: Gmail thread linkage + talon backup | High | Low | Per-thread + per-message via metadata | Highest — two independent signals | Lowest |

The hybrid wins on every dimension except simplicity. Per-thread indexing wins the simplicity column and is "close enough" on every other dimension. For a personal archive (not a high-stakes production system), per-thread is the right starting point; the hybrid is the upgrade path if grep precision turns out to be a problem.

## Why per-thread + Gmail-aware stripping beats talon alone

Gmail's API gives you `threadId`, ordered `messages[]`, and `internalDate` per message. You **already know** what every prior message contained — so the quoted block in message N is, by definition, a subset of messages 1..N-1's content. You don't need to parse the quoted block to know it's redundant; you need to know where it starts so you can cut it off. That's a much easier problem than talon's "given an arbitrary email, separate quote from reply": a regex pass for the attribution line (`On ... wrote:`, `Le ... a écrit :`, `Am ... schrieb`) plus an HTML pass for `div.gmail_quote` / `blockquote[type="cite"]` will catch >95% of cases, because you only need to find the *start* of the quote, not classify its contents. mail-parser-reply already implements the attribution-line regexes across 13 languages.

## POC

```python
# pip install mail-parser-reply beautifulsoup4 lxml
from mailparser_reply import EmailReplyParser
from bs4 import BeautifulSoup

SAMPLE_TEXT = """\
Sounds good, ship it Friday.

Best,
Alice
Sent from my iPhone

On Mon, Jan 15, 2024 at 09:00, Bob <bob@example.com> wrote:
> Can we push the launch to Friday? QA found one more issue.
>
> --
> Bob Smith
> Eng Manager
>
> On Sun, Jan 14, 2024 at 18:22, Alice <alice@example.com> wrote:
>> Friday or Monday — which do you prefer?
"""

SAMPLE_HTML = """\
<div dir="ltr">Sounds good, ship it Friday.<br><br>
  <div class="gmail_signature">Alice<br>Sent from my iPhone</div>
</div>
<div class="gmail_quote">
  On Mon, Jan 15, 2024 at 09:00, Bob &lt;bob@example.com&gt; wrote:
  <blockquote class="gmail_quote" style="margin:0 0 0 .8ex;border-left:1px #ccc solid;padding-left:1ex">
    Can we push the launch to Friday?
  </blockquote>
</div>
"""

# Plaintext: mail-parser-reply gives us per-reply splits + signature detection
msg = EmailReplyParser(languages=["en", "de", "fr"]).read(text=SAMPLE_TEXT)
print("Latest reply only:")
print(msg.replies[0].body)
# -> "Sounds good, ship it Friday."

print("Detected signatures:", msg.replies[0].signatures)
# -> ["Best,\nAlice\nSent from my iPhone"]

# HTML: drop the quote container before extracting text
soup = BeautifulSoup(SAMPLE_HTML, "lxml")
for sel in ["div.gmail_quote", "blockquote[type=cite]",
            "div#divRplyFwdMsg", "div#appendonsend"]:
    for node in soup.select(sel):
        node.decompose()
print("HTML new content only:")
print(soup.get_text("\n", strip=True))
# -> "Sounds good, ship it Friday.\nAlice\nSent from my iPhone"
```

## Known failure modes no library will catch

1. **Inline replies above the quote.** Some users (especially engineers) interleave their reply *inside* the quoted block: `> What time?` followed by `2pm` on the next line, with no attribution. Every heuristic stripper deletes the `2pm` along with the quote.
2. **"Ghost" plaintext from forwarded HTML.** Outlook frequently sends a plaintext alternative that has lost its quote markers entirely — the quoted content appears as ordinary paragraphs. Stripping the HTML side does nothing for the text side. Prefer HTML when both are present.
3. **Top-posted with no attribution.** Pasted content from another email or doc, no `On ... wrote:` header. Indistinguishable from the user's own writing.
4. **Disclaimers vs. signatures.** Corporate "this email is confidential" blocks are often longer than the message itself. mail-parser-reply has a `disclaimers` field that helps; talon does not.
5. **Encoded/MIME-mangled threads.** Long `=?utf-8?Q?...?=` lines or `quoted-printable` soft-wraps can split the attribution header across "lines" and defeat the regex. Normalize encoding *before* stripping.
6. **Auto-replies and ticket-system footers** ("Ref: #INC-12345 — please reply above this line"). Add a small denylist of these once you see them in your own archive.
