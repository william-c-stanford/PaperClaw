"""Convert a Markdown research paper to HTML and PDF.

Uses markdown2 for Markdown → HTML and weasyprint for HTML → PDF.
If weasyprint is not available the PDF step is silently skipped and
only the Markdown file is saved (the route then serves that instead).
"""

import re
from pathlib import Path

_CSS = """
@page {
  size: letter;
  margin: 1in;
  @bottom-center {
    content: counter(page);
    font-family: serif;
    font-size: 10pt;
    color: #555;
  }
}

body {
  font-family: 'Times New Roman', Times, serif;
  font-size: 11pt;
  line-height: 1.65;
  color: #111;
  max-width: 6.5in;
  margin: 0 auto;
}

h1 {
  font-size: 17pt;
  font-weight: bold;
  text-align: center;
  margin: 0 0 6pt;
  line-height: 1.3;
}

.authors {
  text-align: center;
  color: #444;
  margin: 4pt 0;
}

.date {
  text-align: center;
  color: #777;
  margin: 4pt 0;
  font-size: 10pt;
}

h2 {
  font-size: 13pt;
  font-weight: bold;
  margin: 18pt 0 6pt;
  border-bottom: 1px solid #ccc;
  padding-bottom: 3pt;
}

h3 {
  font-size: 11pt;
  font-weight: bold;
  margin: 12pt 0 4pt;
}

h4 {
  font-size: 11pt;
  font-weight: bold;
  font-style: italic;
  margin: 8pt 0 4pt;
}

p {
  margin: 6pt 0;
  text-align: justify;
  hyphens: auto;
}

ul, ol {
  margin: 6pt 0;
  padding-left: 18pt;
}

li {
  margin: 3pt 0;
}

pre {
  background: #f6f6f6;
  border: 1px solid #ddd;
  border-radius: 4pt;
  padding: 8pt;
  font-size: 9pt;
  line-height: 1.4;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}

code {
  font-family: 'Courier New', Courier, monospace;
  font-size: 9.5pt;
  background: #f0f0f0;
  padding: 1pt 3pt;
  border-radius: 2pt;
}

pre code {
  background: none;
  padding: 0;
}

table {
  border-collapse: collapse;
  width: 100%;
  margin: 10pt 0;
  font-size: 10pt;
}

th, td {
  border: 1px solid #bbb;
  padding: 4pt 8pt;
  text-align: left;
}

th {
  background: #efefef;
  font-weight: bold;
}

blockquote {
  border-left: 3pt solid #999;
  padding-left: 10pt;
  color: #555;
  margin: 8pt 0;
}

hr {
  border: none;
  border-top: 1px solid #ccc;
  margin: 14pt 0;
}

strong { color: #000; }
em { font-style: italic; }

a { color: #1a56db; }

img.math-inline {
  vertical-align: middle;
  height: 1.05em;
}

img.math-display {
  display: block;
  margin: 10pt auto;
  max-width: 90%;
}

.math-fallback {
  font-family: 'Courier New', Courier, monospace;
  font-size: 10pt;
  color: #222;
}

div.math-fallback {
  text-align: center;
  margin: 8pt 0;
}
"""


_DISPLAY_MATH_RE = re.compile(r"\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]")
_INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\s)([^\n$]+?)(?<!\\)\$|\\\(([\s\S]+?)\\\)")


def _latex_to_img(tex: str, display: bool) -> str:
    """Rasterize a LaTeX fragment to a base64 PNG <img> via matplotlib mathtext.

    WeasyPrint cannot lay out MathML, so equations are rendered to images.
    On any failure (unsupported LaTeX, missing matplotlib) we fall back to a
    clean styled source span — the PDF never shows raw \\[ \\] delimiters."""
    tex = tex.strip()
    try:
        import base64
        from io import BytesIO

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fontsize = 15 if display else 12
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0)
        fig.text(0, 0, f"${tex}$", fontsize=fontsize)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200, transparent=True,
                    bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        data = base64.b64encode(buf.getvalue()).decode()
        cls = "math-display" if display else "math-inline"
        return f'<img class="{cls}" src="data:image/png;base64,{data}" alt="{tex}" />'
    except Exception:
        tag = "div" if display else "span"
        return f'<{tag} class="math-fallback">{tex}</{tag}>'


def _render_math(md_text: str) -> str:
    """Replace LaTeX math delimiters with rendered equation images."""
    def _disp(m: re.Match) -> str:
        return _latex_to_img(m.group(1) or m.group(2) or "", display=True)

    def _inl(m: re.Match) -> str:
        return _latex_to_img(m.group(1) or m.group(2) or "", display=False)

    text = _DISPLAY_MATH_RE.sub(_disp, md_text)
    text = _INLINE_MATH_RE.sub(_inl, text)
    return text


def _md_to_html_body(md_text: str) -> str:
    import markdown2
    # Render math to MathML first so markdown2 doesn't mangle LaTeX, and so
    # the generated PDF shows typeset equations rather than \[ \] delimiters.
    md_text = _render_math(md_text)
    return markdown2.markdown(
        md_text,
        extras=["tables", "fenced-code-blocks", "break-on-newline"],
    )


def _build_html(md_text: str) -> str:
    body = _md_to_html_body(md_text)
    # Style the Authors / Date metadata lines
    body = re.sub(
        r'<p><strong>Authors:</strong>(.*?)</p>',
        r'<p class="authors"><strong>Authors:</strong>\1</p>',
        body,
    )
    body = re.sub(
        r'<p><strong>Date:</strong>(.*?)</p>',
        r'<p class="date">\1</p>',
        body,
    )
    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        f'<style>{_CSS}</style>\n'
        '</head>\n'
        f'<body>{body}</body>\n'
        '</html>'
    )


def markdown_to_pdf(md_text: str, output_path: Path) -> None:
    """Render Markdown → HTML → PDF via weasyprint. Raises on failure."""
    from weasyprint import HTML
    html = _build_html(md_text)
    HTML(string=html).write_pdf(str(output_path))


def markdown_to_html(md_text: str) -> str:
    return _build_html(md_text)
