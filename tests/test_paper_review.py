"""Tests for the deterministic paper-compliance review (compile mocked — no LaTeX)."""

from paperclaw import paper_review


def _fake_compile(pages, log=""):
    return lambda work_dir, tex_name="paper.tex": (True, pages, log)


def test_review_flags_disallowed_packages_and_commands(tmp_path, monkeypatch):
    (tmp_path / "venue").mkdir()
    (tmp_path / "paper.tex").write_text(
        r"\documentclass[letterpaper]{article}"
        "\n\\usepackage[submission]{aaai2026}"
        "\n\\usepackage[margin=1in]{geometry}"
        "\n\\usepackage{hyperref}"
        "\n\\begin{document}\\newpage abstract references\\end{document}\n",
        encoding="utf-8")
    monkeypatch.setattr(paper_review, "compile_tex", _fake_compile(5))
    rep = paper_review.review_paper(tmp_path, "paper.tex", page_limit=7, venue_style=True)
    msgs = " ".join(m for _, m in rep["issues"])
    assert rep["ok"] is False
    assert "geometry" in msgs and "hyperref" in msgs and r"\newpage" in msgs


def test_review_page_limit(tmp_path, monkeypatch):
    (tmp_path / "paper.tex").write_text(
        r"\documentclass{article}\begin{document}abstract references\end{document}",
        encoding="utf-8")
    monkeypatch.setattr(paper_review, "compile_tex", _fake_compile(9))
    rep = paper_review.review_paper(tmp_path, "paper.tex", page_limit=7, venue_style=False)
    assert rep["ok"] is False
    assert any("exceeds the 7-page" in m for _, m in rep["issues"])


def test_review_compliant(tmp_path, monkeypatch):
    (tmp_path / "paper.tex").write_text(
        r"\documentclass[letterpaper]{article}\usepackage[submission]{aaai2026}"
        r"\begin{document}\section{Abstract}x \bibliography{ref}\end{document}",
        encoding="utf-8")
    # A clean compile log + a PDF with the expected structure words.
    monkeypatch.setattr(paper_review, "compile_tex", _fake_compile(6))
    monkeypatch.setattr(paper_review, "_pdf_pages_text",
                        lambda p: ["Abstract intro method", "results", "References\n[1] x"])
    rep = paper_review.review_paper(tmp_path, "paper.tex", page_limit=7, venue_style=True)
    assert rep["ok"] is True
    assert "COMPLIANT" in paper_review.format_report(rep)


def test_page_limit_counts_main_text_only(tmp_path, monkeypatch):
    """A 9-page PDF whose references/appendix fill p.8-9 is 7 main-text pages → OK."""
    (tmp_path / "paper.tex").write_text(
        r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    monkeypatch.setattr(paper_review, "compile_tex", _fake_compile(9))
    pages_text = ["Body"] * 7 + ["References\n[1] ...", "Appendix\nA Proofs"]
    monkeypatch.setattr(paper_review, "_pdf_pages_text", lambda p: pages_text)
    rep = paper_review.review_paper(tmp_path, "paper.tex", page_limit=7, venue_style=False)
    assert rep["main_pages"] == 7 and rep["extra_start"] == 8
    assert not any("exceeds" in m for _, m in rep["issues"])
    assert "main text" in paper_review.format_report(rep)  # header shows the distinction


def test_page_limit_counts_shared_boundary_page(tmp_path, monkeypatch):
    """When references begin MID-PAGE (right after the conclusion), the main text
    shares that page, so it still counts toward the limit."""
    (tmp_path / "paper.tex").write_text(
        r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    monkeypatch.setattr(paper_review, "compile_tex", _fake_compile(6))
    # p.6: a full page of Conclusion prose, THEN References lower down → main = 6.
    shared = "Conclusion. " + ("the method generalizes well. " * 40) + "\nReferences\n[1] x"
    monkeypatch.setattr(paper_review, "_pdf_pages_text", lambda p: ["Body"] * 5 + [shared])
    rep = paper_review.review_paper(tmp_path, "paper.tex", page_limit=7, venue_style=False)
    assert rep["main_pages"] == 6 and rep["extra_start"] == 6


def test_page_limit_flags_when_main_text_over(tmp_path, monkeypatch):
    """References starting on p.9 → 8 main-text pages → over a 7-page limit."""
    (tmp_path / "paper.tex").write_text(
        r"\documentclass{article}\begin{document}x\end{document}", encoding="utf-8")
    monkeypatch.setattr(paper_review, "compile_tex", _fake_compile(9))
    monkeypatch.setattr(paper_review, "_pdf_pages_text",
                        lambda p: ["Body"] * 8 + ["References\n[1] x"])
    rep = paper_review.review_paper(tmp_path, "paper.tex", page_limit=7, venue_style=False)
    assert rep["main_pages"] == 8 and rep["ok"] is False
    assert any("exceeds the 7-page" in m and "begin on p.9" in m for _, m in rep["issues"])


def test_review_compile_failure_reports_lint_too(tmp_path, monkeypatch):
    """Even when the compile fails, the static lint (disallowed package) is reported."""
    (tmp_path / "paper.tex").write_text(
        r"\documentclass{article}\usepackage{fullpage}\begin{document}x\end{document}",
        encoding="utf-8")
    monkeypatch.setattr(paper_review, "compile_tex",
                        lambda wd, tn="paper.tex": (False, None, "! LaTeX Error: File `fullpage.sty' not found."))
    rep = paper_review.review_paper(tmp_path, "paper.tex", venue_style=True)
    msgs = " ".join(m for _, m in rep["issues"])
    assert rep["ok"] is False
    assert "did not compile" in msgs and "fullpage" in msgs


def test_page_limit_from_style():
    assert paper_review.page_limit_from_style("Review submissions may contain up to 7 pages") == 7
    assert paper_review.page_limit_from_style("up to 9 pages of content") == 9
    assert paper_review.page_limit_from_style(None) == paper_review.DEFAULT_PAGE_LIMIT
