"""Unit tests for `pi_email.html_strip.clean_html_and_css`.

The cleaner is the chokepoint that runs on every Gmail body BEFORE the entity
extractor sees it. Each test case mirrors one of the failure modes that
produced a false-positive entity in a real Gmail run.

Specifically, the regression that triggered this module was:
  - HTML email signatures with `style="font-family: Arial, 'Helvetica Neue'"`
    surviving as naked tokens after the previous regex tag strip, then
    landing as proper-noun entities like `[[people/arial]]` and
    `[[people/helvetica-neue]]` in the materialized profile.

We assert that:
  - The bug-fixing transforms (steps 1-5) actually remove these tokens.
  - Body prose containing the SAME words but not in CSS context (test 9)
    is left untouched — the cleaner is conservative on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

POC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(POC_ROOT / "src"))

from pi_email.html_strip import clean_html_and_css  # noqa: E402


# ---------------- Cases 1-7: targeted transforms ----------------


def test_plain_text_unchanged():
    """Pure plaintext passes through unchanged (modulo whitespace norm)."""
    text = "Hi Alice,\n\nSee you Saturday.\n\n- Bob"
    assert clean_html_and_css(text) == text


def test_basic_html_tag_strip():
    """`<p>Hello <b>world</b></p>` -> `Hello world`."""
    assert clean_html_and_css("<p>Hello <b>world</b></p>") == "Hello world"


def test_style_block_with_contents_removed():
    """`<style>.foo { color: red; }</style>Hello` -> `Hello`."""
    src = "<style>.foo { color: red; }</style>Hello"
    assert clean_html_and_css(src) == "Hello"


def test_script_block_with_contents_removed():
    """`<script>alert(1)</script>Hello` -> `Hello`."""
    src = "<script>alert(1)</script>Hello"
    assert clean_html_and_css(src) == "Hello"


def test_inline_font_family_rule_stripped_from_prose():
    """Critical case: inline `font-family: Arial, ...;` declaration stripped
    out of running text, but surrounding prose preserved.

    This is the specific shape we saw in a real Gmail body — a plaintext
    line that happened to contain a CSS rule literally."""
    src = 'Some text. font-family: Arial, "Helvetica Neue", sans-serif; more text.'
    out = clean_html_and_css(src)
    # The font names should be gone.
    assert "Arial" not in out
    assert "Helvetica" not in out
    assert "sans-serif" not in out
    # Surrounding prose preserved.
    assert "Some text." in out
    assert "more text." in out


def test_html_entities_decoded():
    """`&amp;` -> `&`, `&#39;` -> `'`, etc."""
    src = "Hi! &amp; goodbye. &#39;quoted&#39; &nbsp;and&nbsp;spaced"
    out = clean_html_and_css(src)
    assert "&amp;" not in out
    assert "&" in out  # The decoded ampersand
    assert "'quoted'" in out


def test_style_attr_on_div_removed():
    """`<div style="font-family: Arial, Helvetica Neue;">Hi</div>` -> `Hi`.

    Step 2 (tag strip) handles this: the style attribute is part of the
    opening tag, so it goes with the tag. After tag strip, only "Hi"
    is left."""
    src = '<div style="font-family: Arial, Helvetica Neue;">Hi</div>'
    out = clean_html_and_css(src)
    assert out == "Hi"
    assert "Arial" not in out
    assert "Helvetica" not in out


# ---------------- Case 8: realistic mail signature ----------------


def test_realistic_html_signature_loses_font_stack():
    """A ~20-line HTML signature with mixed tags + the specific font-stack
    we know is in the corpus should NOT leave Arial / Helvetica / Segoe in
    the output."""
    signature = """\
<div style="font-family: Arial, 'Helvetica Neue', 'Segoe UI', sans-serif; font-size: 14px; color: #333333;">
  <p style="margin: 0 0 10px 0;">Hey Alice,</p>
  <p style="margin: 0 0 10px 0;">Quick update on the family reunion plans.</p>
  <p style="margin: 0 0 10px 0;">Let me know what works.</p>
  <p style="margin: 0;">-- </p>
  <table cellpadding="0" cellspacing="0" border="0" style="font-family: Arial, Helvetica Neue, Segoe UI;">
    <tr>
      <td style="padding-right: 12px; border-right: 1px solid #cccccc;">
        <img src="https://example.com/sig.png" width="60" height="60" style="border-radius: 30px;" />
      </td>
      <td style="padding-left: 12px;">
        <div style="font-weight: bold; color: rgb(34, 34, 34); font-family: Arial, 'Helvetica Neue', sans-serif;">Bob Smith</div>
        <div style="color: #666666;">Project Lead</div>
        <div><a href="mailto:bob@example.com" style="color: #1a73e8; text-decoration: none;">bob@example.com</a></div>
        <div><a href="https://example.com" style="color: #1a73e8;">example.com</a></div>
      </td>
    </tr>
  </table>
</div>
"""
    out = clean_html_and_css(signature)
    # The bug being fixed: no font-family value tokens should survive.
    assert "Arial" not in out, f"'Arial' survived: {out!r}"
    assert "Helvetica" not in out, f"'Helvetica' survived: {out!r}"
    assert "Segoe" not in out, f"'Segoe' survived: {out!r}"
    # Sanity: the actual body content should still be present.
    assert "Hey Alice" in out
    assert "Bob Smith" in out
    assert "Project Lead" in out
    assert "bob@example.com" in out


# ---------------- Case 9: false-positive guard ----------------


def test_body_prose_with_font_names_preserved():
    """Body prose that legitimately mentions Helvetica or Arial as topical
    references (not as CSS values) must survive unchanged.

    The cleaner only strips when the names appear in `font-family: ...;`
    context — a sentence like "I prefer Helvetica over Arial." stays
    intact."""
    src = "I prefer Helvetica over Arial."
    out = clean_html_and_css(src)
    assert "Helvetica" in out
    assert "Arial" in out
    assert out == src


# ---------------- Case 10: whitespace normalization ----------------


def test_multiple_consecutive_newlines_collapsed():
    """3+ consecutive newlines collapse to exactly 2 (one blank line)."""
    src = "Line one.\n\n\n\n\nLine two."
    out = clean_html_and_css(src)
    assert out == "Line one.\n\nLine two."


# ---------------- Extra coverage / edge cases ----------------


def test_empty_input_returns_empty():
    assert clean_html_and_css("") == ""


def test_hex_color_stripped():
    """Standalone hex color literals are removed."""
    src = "color is #ff0000 here"
    out = clean_html_and_css(src)
    assert "#ff0000" not in out


def test_word_with_hash_prefix_kept():
    """A `#topic` style hashtag (non-hex chars) is NOT mistaken for a color."""
    src = "Discussing #topic today."
    out = clean_html_and_css(src)
    assert "#topic" in out


def test_unsubscribe_on_own_line_stripped():
    """`Unsubscribe` on a short standalone line is treated as footer noise."""
    src = "Read more.\n\nUnsubscribe\n\nPrivacy Policy"
    out = clean_html_and_css(src)
    assert "Unsubscribe" not in out
    assert "Privacy Policy" not in out
    assert "Read more." in out


def test_unsubscribe_inside_sentence_kept():
    """The word 'unsubscribe' inside body prose stays."""
    src = "If you wish to unsubscribe from this list, click here."
    out = clean_html_and_css(src)
    assert "unsubscribe" in out
