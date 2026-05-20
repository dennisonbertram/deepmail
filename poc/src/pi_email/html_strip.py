"""Strip HTML tags and CSS noise from email bodies before downstream processing.

The motivation is a real failure mode we hit on a Gmail run: an HTML email
signature (or even a plaintext signature that copied CSS literally) contained
`style="font-family: Arial, 'Helvetica Neue', 'Segoe UI'"`. After our previous
regex tag strip, the values "Arial", "Helvetica Neue", "Segoe UI" survived as
naked tokens in the body and got tagged as proper-noun entities by the spaCy
NER pipeline — landing in the materialized profile as family members.

This module is a slightly-more-careful cleaner that runs BEFORE
`strip_quotes_and_signatures`. It is intentionally pure-stdlib (`re`, `html`)
to avoid pulling `beautifulsoup4` / `html2text` just for this case. The
rules are conservative: we err toward leaving body prose alone (so "I
prefer Helvetica over Arial" survives untouched) and only strip when
text matches an obvious CSS-rule shape (`<property>: <value>;`).

Order of operations matters:
  1. Strip <script>...</script> and <style>...</style> blocks with contents.
  2. Strip remaining HTML tags (preserve their text content).
  3. Decode HTML entities (`&amp;` -> `&`, etc.).
  4. Strip surviving CSS declarations (the actual source of the bug).
  5. Strip hex / rgb / rgba color literals.
  6. Strip footer boilerplate ("Unsubscribe", "View this email in your
     browser", "Privacy Policy") when it appears on its own short line.
  7. Collapse whitespace and runs of blank lines.
"""

from __future__ import annotations

import html
import re


# --- Step 1: <script> / <style> blocks --------------------------------------

# DOTALL so `.` matches newlines (these blocks can span many lines).
# IGNORECASE because HTML tag names are case-insensitive.
_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.DOTALL | re.IGNORECASE)
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.DOTALL | re.IGNORECASE)

# Also handle the (rarer) self-terminating variants and the (broken-html)
# case where the opening tag is present but the closing is missing. We use
# DOTALL again and a non-greedy match so adjacent valid HTML isn't eaten.
_SCRIPT_OPEN_RE = re.compile(r"<script\b[^>]*/>", re.IGNORECASE)
_STYLE_OPEN_RE = re.compile(r"<style\b[^>]*/>", re.IGNORECASE)


# --- Step 2: remaining HTML tags --------------------------------------------

# Strip any remaining tag like `<p>`, `<div class="x">`, `</span>`, `<br/>`.
# Replace with a single space so adjacent words don't fuse (`<b>foo</b>bar`
# would otherwise collapse to "foobar").
_TAG_RE = re.compile(r"<[^>]+>")


# --- Step 4: CSS declarations -----------------------------------------------

# The specific properties we always strip (these are the ones email
# signatures tend to leak as standalone text). Each pattern matches
# "<prop>:<value>;" with optional trailing semicolon and optional space.
_CSS_PROP_NAMES = (
    "font-family",
    "font-size",
    "font-weight",
    "font-style",
    "color",
    "background",
    "background-color",
    "background-image",
    "padding",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "margin",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "border",
    "border-top",
    "border-right",
    "border-bottom",
    "border-left",
    "border-radius",
    "border-color",
    "border-style",
    "border-width",
    "line-height",
    "text-align",
    "text-decoration",
    "text-transform",
    "letter-spacing",
    "vertical-align",
    "display",
    "width",
    "height",
    "max-width",
    "min-width",
    "max-height",
    "min-height",
)
_CSS_KNOWN_PROP_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in _CSS_PROP_NAMES) + r")\s*:\s*[^;\n]+;?",
    re.IGNORECASE,
)

# Generic catchall for a single CSS rule on its own line — must have a
# semicolon, must be the only thing on the line. We do NOT use this
# anywhere except inside a multi-rule line because it's too aggressive
# (would eat URLs like "https://..." that look like `https:`).
_CSS_LINE_RE = re.compile(
    r"^[ \t]*[a-z-]+[ \t]*:[ \t]*[^;\n]+;[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


# --- Step 5: color literals -------------------------------------------------

# `#abc`, `#abc123`, `#abcdef12` — 3, 4, 6, or 8 hex digits with word
# boundaries on both sides so URL fragments like `#section` aren't hit
# (those have letters that aren't all-hex). We additionally require the
# whole token to be hex digits.
_HEX_COLOR_RE = re.compile(r"(?<![\w])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})(?![\w])")

# `rgb(255, 0, 0)`, `rgba(0,0,0,0.5)`, `hsl(...)`, `hsla(...)` — just
# match the function-call shape.
_RGB_RE = re.compile(r"\b(?:rgba?|hsla?)\s*\([^)]*\)", re.IGNORECASE)


# --- Step 6: footer boilerplate ---------------------------------------------

# These are short stand-alone lines we treat as footer noise. The line
# must be reasonably short — if "Unsubscribe" appears inside a sentence
# we leave it alone. The `^...$` anchors with MULTILINE handle that.
_FOOTER_PHRASES = (
    r"view (?:this )?(?:email|message) in (?:your |a )?browser",
    r"if you (?:no longer wish|don't want) to receive these emails",
    r"unsubscribe",
    r"privacy policy",
    r"manage (?:your )?preferences",
    r"update (?:your )?(?:email )?preferences",
    r"add us to your address book",
    r"this email was sent to .+",
)
_FOOTER_LINE_RE = re.compile(
    r"^[ \t]*(?:" + "|".join(_FOOTER_PHRASES) + r")[ \t.!]*$",
    re.MULTILINE | re.IGNORECASE,
)


# --- Step 7: whitespace normalization ---------------------------------------

# Collapse runs of horizontal whitespace.
_HWS_RE = re.compile(r"[ \t\f\v]+")
# Collapse 3+ newlines (with optional whitespace) to exactly two.
_BLANK_LINES_RE = re.compile(r"(?:[ \t]*\n){3,}")


def clean_html_and_css(text: str) -> str:
    """Strip HTML tags, CSS declarations, and signature font-stack noise.

    Input may be HTML, plaintext-with-leaked-CSS, or genuinely clean
    plaintext. Output is plaintext with HTML/CSS noise removed.

    Conservative by design: we only strip text that looks like an HTML
    tag or a CSS rule. Body prose that legitimately mentions font names
    ("I prefer Helvetica over Arial") is left intact because those words
    don't appear in `font-family: ...;` context.
    """
    if not text:
        return ""

    # Step 1: kill <script> and <style> blocks with contents.
    out = _SCRIPT_BLOCK_RE.sub(" ", text)
    out = _STYLE_BLOCK_RE.sub(" ", out)
    out = _SCRIPT_OPEN_RE.sub(" ", out)
    out = _STYLE_OPEN_RE.sub(" ", out)

    # Step 2: strip remaining tags. Replace with a space so "<b>x</b>y"
    # doesn't fuse to "xy".
    out = _TAG_RE.sub(" ", out)

    # Step 3: decode entities. stdlib handles &amp; &lt; &gt; &quot;
    # &nbsp; &#39; &hellip; and the long tail.
    out = html.unescape(out)

    # Step 4: kill CSS declarations that survived (the main bug). We do
    # the known-properties pass first (matches inline single rules even
    # when embedded in a sentence) and the line-anchored catchall second.
    out = _CSS_KNOWN_PROP_RE.sub(" ", out)
    out = _CSS_LINE_RE.sub("", out)

    # Step 5: color literals — these often survive as leftover after a
    # CSS rule is partially stripped.
    out = _HEX_COLOR_RE.sub(" ", out)
    out = _RGB_RE.sub(" ", out)

    # Step 6: footer boilerplate on standalone short lines.
    out = _FOOTER_LINE_RE.sub("", out)

    # Step 7: whitespace cleanup. Horizontal first, then blank-line
    # collapse. We leave a final `.strip()` to remove the leading and
    # trailing whitespace that the substitutions created.
    out = _HWS_RE.sub(" ", out)
    out = _BLANK_LINES_RE.sub("\n\n", out)
    # Strip trailing/leading whitespace on each line individually so the
    # ` ` we substituted for stripped tags doesn't show up as a leading
    # indent.
    lines = [ln.strip() for ln in out.split("\n")]
    out = "\n".join(lines)
    # One more blank-line collapse since per-line stripping may have
    # created additional empty lines.
    out = _BLANK_LINES_RE.sub("\n\n", out)
    return out.strip()
