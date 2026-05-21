"""Markdown corpus: read raw email markdown files from disk into structured records.

A corpus record is one message: YAML frontmatter (from/to/subject/date/message_id/
thread_id) + plaintext body. The body is read as-is; quote stripping happens in
strip_quotes.py at the point of entity extraction, so the original file is preserved
for re-derivation (per research/04 "keep the raw alongside the cleaned corpus").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


MessageId = str


@dataclass
class Message:
    """One email message: headers from frontmatter + body."""

    message_id: MessageId
    thread_id: str
    from_addr: str
    to_addr: str
    subject: str
    date: str
    body: str
    source_path: Path
    # Body after reply-quote stripping; populated lazily by callers.
    body_clean: str | None = None
    # Raw header map (case-preserved keys). Populated by GmailSearcher so the
    # bulk-mail filter can inspect `List-Unsubscribe`, `Precedence`, etc.
    # Fixture-loaded messages leave it empty — they go through the same
    # filter but only the From: heuristic can fire on them.
    headers: dict[str, str] = field(default_factory=dict)
    # True if `filters.is_bulk_message(headers, from_addr)` flagged this as
    # marketing / newsletter / list-managed mail. The message still lands
    # in the corpus (so we can cite it later) but downstream entity
    # extraction skips it — newsletters dump too many capitalized phrases
    # to be trustworthy entity sources.
    is_bulk: bool = False
    # Persons extracted from a Google Calendar notification email (Pass 17A).
    # Populated by `gmail_searcher._message_from_payload` when the sender
    # matches `is_calendar_notification_sender`. Each entry is a
    # `calendar_email_parser.CalendarEmailPerson` with name + optional
    # email + family signal strength. Typed as `list` (not `list[
    # CalendarEmailPerson]`) to avoid a circular import between
    # `corpus.py` and `calendar_email_parser.py`; downstream callers
    # `cast` / treat each element as `CalendarEmailPerson`.
    calendar_persons: list = field(default_factory=list)
    # Email address of the Gmail account this message was fetched from.
    # Set by MultiAccountSearcher when merging results from multiple
    # accounts. Empty string for single-account / fixture-loaded messages.
    source_account: str = ""

    def all_text(self) -> str:
        """Concatenated headers + body — what searchers grep against."""
        return f"{self.subject}\n{self.from_addr}\n{self.to_addr}\n{self.body}"

    def clean_text(self) -> str:
        """All-text using the stripped body if available, raw body otherwise."""
        body = self.body_clean if self.body_clean is not None else self.body
        return f"{self.subject}\n{self.from_addr}\n{self.to_addr}\n{body}"


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parse_message_file(path: Path) -> Message:
    """Parse one fixture markdown file into a Message.

    Format: YAML frontmatter delimited by --- lines, then a plaintext body.

    We deliberately do NOT call `html_strip.clean_html_and_css` here because
    fixtures are hand-authored plaintext markdown — they don't carry HTML
    tags or CSS rules. The cleaner runs on real Gmail bodies inside
    `gmail_searcher.py` (where signatures and inline styles actually
    appear). If a future fixture ever needs HTML stripping we can wire it
    in, but adding it here today would just slow the fixture loader.
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {path}")
    fm = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    return Message(
        message_id=str(fm.get("message_id") or path.stem),
        thread_id=str(fm.get("thread_id") or fm.get("message_id") or path.stem),
        from_addr=str(fm.get("from", "")),
        to_addr=str(fm.get("to", "")),
        subject=str(fm.get("subject", "")),
        date=str(fm.get("date", "")),
        body=body,
        source_path=path,
    )


@dataclass
class Corpus:
    """An in-memory collection of Messages, addressed by message_id."""

    messages: dict[MessageId, Message] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, root: Path) -> "Corpus":
        """Load every *.md file under root as a Message."""
        c = cls()
        for path in sorted(root.glob("*.md")):
            msg = parse_message_file(path)
            c.messages[msg.message_id] = msg
        return c

    def add(self, msg: Message) -> bool:
        """Add a message. Returns True if new, False if already present."""
        if msg.message_id in self.messages:
            return False
        self.messages[msg.message_id] = msg
        return True

    def list_ids(self) -> list[MessageId]:
        return list(self.messages.keys())

    def get(self, msg_id: MessageId) -> Message | None:
        return self.messages.get(msg_id)

    def fingerprint(self) -> str:
        """sha256 of the sorted message-id list. Per research/04, this is the
        cheap staleness check for derived profiles."""
        h = hashlib.sha256()
        for mid in sorted(self.messages.keys()):
            h.update(mid.encode("utf-8"))
            h.update(b"\n")
        return "sha256:" + h.hexdigest()

    def __len__(self) -> int:
        return len(self.messages)
