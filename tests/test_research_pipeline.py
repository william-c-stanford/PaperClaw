"""Tests for research_pipeline parsing helpers."""

from paperclaw.research_pipeline import _extract_block, _extract_paper

# A realistic paper with a NESTED ``` code fence in Methodology — the case that
# truncated papers at Methodology when parsed with the generic block extractor.
PAPER_WITH_CODE = """\
Here is the paper.

```paper
# Spectral Forecasting

## Abstract
We propose a method.

## 3. Methodology
The algorithm:

```
for t in range(T):
    x = f(x)
```

The above runs in O(T).

## 7. Conclusion
We conclude with strong results.

## References
1. Someone (2024). A paper. (verify)
```
"""


def test_extract_paper_keeps_content_after_nested_fence():
    out = _extract_paper(PAPER_WITH_CODE)
    # The whole paper survives — Conclusion and References are NOT truncated.
    assert "## 7. Conclusion" in out
    assert "## References" in out
    # The nested code block is preserved verbatim.
    assert "for t in range(T):" in out
    # Leading commentary before the fence is dropped.
    assert "Here is the paper." not in out
    # The paper's own closing fence is removed (no trailing bare ```).
    assert not out.rstrip().endswith("```")


def test_generic_block_extractor_truncates_at_nested_fence():
    # Documents WHY _extract_paper exists: the generic extractor stops early.
    truncated = _extract_block(PAPER_WITH_CODE, "paper")
    assert "## 7. Conclusion" not in truncated


def test_extract_paper_handles_truncated_unclosed_block():
    # Model ran out of tokens — no closing fence. Salvage everything present.
    raw = "```paper\n# Title\n\n## Abstract\nText.\n\n## 3. Methodology\nCut off here"
    out = _extract_paper(raw)
    assert out.startswith("# Title")
    assert out.rstrip().endswith("Cut off here")


def test_extract_paper_without_fence_falls_back():
    raw = "# Title\n\n## Abstract\nNo fence at all."
    out = _extract_paper(raw)
    assert "# Title" in out and "No fence at all." in out
