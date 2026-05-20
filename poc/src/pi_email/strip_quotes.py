"""Strip reply quotes and obvious signatures from a plaintext email body.

Per research/03, the recommended stack is `mail-parser-reply` for plaintext +
`quotequail` for HTML / forward-vs-reply classification. Both are optional —
if the import fails we fall back to a small regex stripper that handles the
most common 'On <date>, <person> wrote:' Gmail-style attribution line and the
`> ` prefix that follows.
"""

from __future__ import annotations

import re


# Best-effort attribution-line detector. Handles the English Gmail format and a
# couple of common variants we see in fixtures. Multilingual variants are handled
# by mail-parser-reply when it's available.
_ATTRIBUTION_RES = [
    re.compile(r"^On .{0,80}wrote:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Le .{0,80}a écrit\s*:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Am .{0,80}schrieb .{0,80}:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Begin forwarded message:\s*$", re.IGNORECASE | re.MULTILINE),
]

# Common mobile signature footers — heuristic, see research/03.
_SIGNATURE_RES = [
    re.compile(r"^Sent from my (iPhone|Android|iPad|Samsung).*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Get Outlook for (iOS|Android).*$", re.IGNORECASE | re.MULTILINE),
]


def _regex_strip(text: str) -> str:
    """Conservative regex fallback. Truncate at the first attribution line, then
    drop quoted (`> ...`) lines, then drop trailing mobile signatures.

    Conservative because false-negative (leaving in a stray quote) costs less than
    false-positive (deleting the user's actual reply). See research/03 §"Known
    failure modes".
    """
    # Truncate at the first attribution we find.
    cut = len(text)
    for r in _ATTRIBUTION_RES:
        m = r.search(text)
        if m:
            cut = min(cut, m.start())
    head = text[:cut]

    # Drop any remaining standalone quoted lines.
    out_lines = []
    for line in head.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(">"):
            continue
        out_lines.append(line)
    result = "\n".join(out_lines)

    # Strip mobile-signature footers.
    for r in _SIGNATURE_RES:
        result = r.sub("", result)

    # Normalize trailing whitespace.
    return result.strip()


def strip_quotes_and_signatures(text: str) -> str:
    """Public entry point. Try mail-parser-reply first, then fall back to regex.

    Returns the cleaned body. Always returns a string; never raises.
    """
    if not text:
        return ""

    # Try mail-parser-reply for proper multilingual attribution detection.
    try:
        from mailparser_reply import EmailReplyParser  # type: ignore

        parsed = EmailReplyParser(languages=["en", "fr", "de"]).read(text=text)
        if parsed and parsed.replies:
            latest = parsed.replies[0].body or ""
            if latest.strip():
                # Apply our signature regex on top — mail-parser-reply's signature
                # detection is good but not exhaustive for mobile auto-appends.
                for r in _SIGNATURE_RES:
                    latest = r.sub("", latest)
                return latest.strip()
    except Exception:
        pass

    return _regex_strip(text)
