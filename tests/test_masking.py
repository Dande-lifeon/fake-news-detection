"""Tests for source-cue masking."""

from __future__ import annotations

import pandas as pd

from fakenews.features.masking import count_hits, mask_body, mask_dataframe


def test_dateline_and_url_removed():
    raw = "WASHINGTON (Reuters) - Hello https://example.com/a world."
    masked = mask_body(raw, "dateline_url")
    assert "(Reuters)" not in masked
    assert "https://example.com" not in masked
    assert "Hello" in masked and "world" in masked


def test_format_mask_removes_media_credits():
    raw = (
        "WATCH: something happened. Photo by Jane Doe/Getty Images. "
        "Follow us on Facebook. Read more: https://t.co/x"
    )
    masked = mask_body(raw, "format")
    assert "Getty" not in masked
    assert "Follow us" not in masked
    assert "Read more" not in masked
    assert "https://" not in masked


def test_mask_dataframe_keeps_title_and_uses_clean_body():
    frame = pd.DataFrame(
        {
            "title": ["A title"],
            "text_raw": ["WASHINGTON (Reuters) - <b>Body</b> text."],
            "text_clean": ["WASHINGTON (Reuters) - Body text."],
            "label": [0],
        }
    )
    masked = mask_dataframe(frame, "dateline_url")
    assert masked.loc[0, "title"] == "A title"
    assert masked.loc[0, "text_clean"] == "WASHINGTON (Reuters) - Body text."
    assert "(Reuters)" not in masked.loc[0, "text_masked"]
    assert "<b>" not in masked.loc[0, "text_masked"]


def test_count_hits_by_label():
    frame = pd.DataFrame(
        {
            "text_raw": [
                "WASHINGTON (Reuters) - real",
                "Fake with https://x.com and Getty Images",
            ],
            "label": [0, 1],
        }
    )
    hits = count_hits(frame)
    assert hits["reuters_dateline"]["real"] == 1
    assert hits["reuters_dateline"]["fake"] == 0
    assert hits["url"]["fake"] == 1
    assert hits["getty"]["fake"] == 1
