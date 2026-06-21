"""Tests for image generation (paper figures) + the deepagents figure tool."""

import base64

import httpx

from paperclaw import images
from paperclaw.config import LLMSettings


def test_not_configured_returns_false(tmp_path):
    s = LLMSettings()  # no image key
    assert images.is_configured(s) is False
    assert images.generate_image(s, "a schematic", tmp_path / "f.png") is False
    assert not (tmp_path / "f.png").exists()


def test_generate_image_b64_success(tmp_path, monkeypatch):
    png = b"\x89PNG\r\n\x1a\nDATA"

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"b64_json": base64.b64encode(png).decode()}]}

    captured = {}

    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw.get("json")
        return _Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    s = LLMSettings(image_api_key="k", image_model="gpt-image-1",
                    image_base_url="https://img.example/v1")
    out = tmp_path / "fig.png"
    assert images.generate_image(s, "a clean schematic", out) is True
    assert out.read_bytes() == png
    assert captured["url"] == "https://img.example/v1/images/generations"
    assert captured["json"]["model"] == "gpt-image-1"


def test_figure_tool_no_image_no_llm(tmp_path):
    """No image API AND no LLM key → the TikZ fallback can't run; degrade gracefully
    with a clear message and write nothing (never a fake/empty figure)."""
    from paperclaw.agents.deep_chat import _make_figure_tool

    tool = _make_figure_tool(LLMSettings(), tmp_path)        # no image key, no LLM key
    msg = tool("a schematic of the method", "fig_method.png")
    assert "image api" in msg.lower() and "no figure" in msg.lower()
    assert not list(tmp_path.glob("*.png"))
    assert not list(tmp_path.glob("figures/*.tex"))


def test_figure_tool_tikz_fallback(tmp_path, monkeypatch):
    """No image API but an LLM IS configured → generate a vector TikZ figure (.tex)
    under figures/ and return \\input guidance."""
    from paperclaw.agents import deep_chat
    from paperclaw.agents.deep_chat import _make_figure_tool
    from paperclaw import llm

    class _R:
        text = ("```latex\n\\begin{tikzpicture}\n\\node[draw](a){Input};\n"
                "\\end{tikzpicture}\n```")

    async def fake_chat(*a, **k):
        return _R()

    monkeypatch.setattr(llm, "chat", fake_chat)
    tool = _make_figure_tool(LLMSettings(api_key="k", provider="openai"), tmp_path)
    msg = tool("a schematic of the method", "fig_method.png")
    assert "tikz" in msg.lower() and "\\input{figures/fig_method.tex}" in msg
    written = (tmp_path / "figures" / "fig_method.tex").read_text()
    assert "tikzpicture" in written and "```" not in written
    assert not list(tmp_path.glob("*.png"))


def test_figure_tool_writes_via_image_api(tmp_path, monkeypatch):
    from paperclaw.agents.deep_chat import _make_figure_tool

    monkeypatch.setattr(images, "generate_image", lambda s, p, out: out.write_bytes(b"PNG") or True)
    tool = _make_figure_tool(LLMSettings(image_api_key="k"), tmp_path)
    msg = tool("a schematic", "fig_method")                  # no extension → .png appended
    assert "Saved figure fig_method.png" in msg
    assert (tmp_path / "fig_method.png").read_bytes() == b"PNG"


def test_download_file_tool(tmp_path, monkeypatch):
    """The bridged download_file tool fetches a binary into the workspace, guarded."""
    from paperclaw.agents.deep_chat import _workspace_tools

    class _R:
        content = b"ZIPDATA"
        def raise_for_status(self): pass

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _R())
    tools = {t.__name__: t for t in _workspace_tools(LLMSettings(), tmp_path)}
    msg = tools["download_file"]("http://x/template.zip", "venue/template.zip")
    assert "Downloaded" in msg
    assert (tmp_path / "venue" / "template.zip").read_bytes() == b"ZIPDATA"
    assert "refused" in tools["download_file"]("http://x", "../escape.zip").lower()
    # the compile + review tools are bridged in too
    assert "compile_latex" in tools and "review_paper" in tools


def test_review_paper_tool(tmp_path, monkeypatch):
    """The bridged review_paper tool runs the compliance review and returns a report."""
    from paperclaw.agents.deep_chat import _workspace_tools
    from paperclaw import paper_review

    (tmp_path / "paper.tex").write_text(r"\documentclass{article}\begin{document}x\end{document}",
                                        encoding="utf-8")
    monkeypatch.setattr(paper_review, "compile_tex", lambda wd, tn="paper.tex": (True, 9, ""))
    tools = {t.__name__: t for t in _workspace_tools(LLMSettings(), tmp_path)}
    out = tools["review_paper"]("paper")  # no .tex suffix → appended
    assert "Review of paper.tex" in out and "9 pages" in out


def test_compile_tex_missing_file(tmp_path):
    """compile_tex returns a clear failure when the .tex is absent (no compiler run)."""
    from paperclaw.iterative_pipeline import compile_tex
    ok, pages, log = compile_tex(tmp_path, "paper.tex")
    assert ok is False and "not found" in log.lower()


def test_pick_latex_engine(monkeypatch):
    """Engine selection matches Overleaf: magic comment > fontspec/unicode-math
    (xe/lua) > pdflatex; downgrades to pdflatex when the chosen engine is absent."""
    import paperclaw.iterative_pipeline as ip
    # All engines present → real selection logic.
    monkeypatch.setattr(ip.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert ip._pick_latex_engine(r"\documentclass{article}") == "pdflatex"
    assert ip._pick_latex_engine(r"\usepackage{fontspec}") == "lualatex"
    assert ip._pick_latex_engine(r"\usepackage[a]{unicode-math}") == "lualatex"
    assert ip._pick_latex_engine("% !TEX program = xelatex\n") == "xelatex"
    assert ip._pick_latex_engine("%!TEX TS-program=lualatex\n") == "lualatex"
    # Chosen engine missing → downgrade to pdflatex.
    monkeypatch.setattr(ip.shutil, "which",
                        lambda name: None if name in ("xelatex", "lualatex") else f"/usr/bin/{name}")
    assert ip._pick_latex_engine(r"\usepackage{fontspec}") == "pdflatex"
    assert ip._pick_latex_engine("% !TEX program = xelatex\n") == "pdflatex"
