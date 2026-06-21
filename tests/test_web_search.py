"""Tests for the web_search + fetch_url tools (parsing, formatting, guards)."""

from pathlib import Path

from paperclaw.tools import fetch_url, web_search

DDG_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=z">Example <b>Title</b></a>
  <a class="result__snippet" href="//x">A snippet about &amp; things.</a>
</div>
<div class="result">
  <a class="result__a" href="https://direct.example.org/">Direct Result</a>
  <a class="result__snippet" href="//y">Second snippet.</a>
</div>
"""


def test_parse_results_decodes_redirect_and_strips():
    results = web_search._parse_results(DDG_HTML, 5)
    assert len(results) == 2
    assert results[0]["title"] == "Example Title"
    assert results[0]["url"] == "https://example.com/page"          # uddg redirect decoded
    assert results[0]["snippet"] == "A snippet about & things."     # tags + entities handled
    assert results[1]["url"] == "https://direct.example.org/"       # plain href untouched


def test_clean_url():
    assert web_search._clean_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2Fb") == "https://a.com/b"
    assert web_search._clean_url("https://plain.com/x") == "https://plain.com/x"


def test_web_search_execute_empty(monkeypatch):
    monkeypatch.setattr(web_search, "_search", lambda q, n: [])
    out = web_search.execute(Path("."), {"query": "nothing here"})
    assert "No web results" in out


def test_web_search_execute_formats(monkeypatch):
    monkeypatch.setattr(web_search, "_search",
                        lambda q, n: [{"title": "T", "url": "https://u", "snippet": "S"}])
    out = web_search.execute(Path("."), {"query": "q"})
    assert "https://u" in out and "T" in out


def test_fetch_url_html_to_text():
    html = "<html><head><style>x{}</style></head><body><script>bad()</script><p>Hello &amp; world</p></body></html>"
    text = fetch_url._html_to_text(html)
    assert "Hello & world" in text
    assert "bad()" not in text and "x{}" not in text


def test_fetch_url_rejects_non_http():
    assert "http/https" in fetch_url.execute(Path("."), {"url": "file:///etc/passwd"})
    assert "required" in fetch_url.execute(Path("."), {"url": ""})
