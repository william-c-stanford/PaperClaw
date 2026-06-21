"""Tests for the writing-style library (module + service + route + CLI)."""

from paperclaw import service, writing_styles
from paperclaw.server.store import Store


def test_builtins_seeded_and_listed(tmp_path):
    styles = writing_styles.list_styles(tmp_path)
    names = {s["name"] for s in styles}
    assert {"technical-concise", "narrative", "formal-theoretical"} <= names
    assert all(s["scope"] == "global" for s in styles)
    # files were written
    assert (tmp_path / "writing_styles" / "narrative.md").is_file()


def test_get_and_name_sanitize(tmp_path):
    md = writing_styles.get_style(tmp_path, None, "technical-concise")
    assert md and md.startswith("# Technical")
    assert writing_styles.get_style(tmp_path, None, "../etc/passwd") is None
    assert writing_styles.get_style(tmp_path, None, "nope") is None


def test_domain_overrides_global(tmp_path):
    domain_dir = tmp_path / "domains" / "d1"
    domain_dir.mkdir(parents=True)
    writing_styles.save_style(tmp_path, domain_dir, "narrative", "# Domain Narrative\ncustom")
    # domain-scoped style wins for the same name
    got = writing_styles.get_style(tmp_path, domain_dir, "narrative")
    assert "Domain Narrative" in got
    listed = {s["name"]: s["scope"] for s in writing_styles.list_styles(tmp_path, domain_dir)}
    assert listed["narrative"] == "domain"


def test_service_and_save(tmp_path):
    store = Store(tmp_path)
    assert len(service.list_writing_styles(store)) >= 3
    saved = service.save_writing_style(store, "My Style!", "# Mine\nx")
    assert saved == "my-style"  # sanitized
    assert service.get_writing_style(store, None, "my-style").startswith("# Mine")


def test_extract_style_arg():
    assert service._extract_style_arg("H1 H2 --style narrative") == ("H1 H2", "narrative")
    assert service._extract_style_arg("--style=technical-concise H3") == ("H3", "technical-concise")
    assert service._extract_style_arg("H1 H2") == ("H1 H2", None)


def test_resolve_writing_style_defaults_to_house_style(tmp_path):
    """No style chosen ⇒ the house DEFAULT_STYLE (prose voice lives in writing styles,
    not the system prompt). A named style replaces it."""
    from paperclaw.prompts.writing_styles import DEFAULT_STYLE
    store = Store(tmp_path)
    assert service.resolve_writing_style(store, None, None) == DEFAULT_STYLE
    assert service.resolve_writing_style(store, None, "") == DEFAULT_STYLE
    chosen = service.resolve_writing_style(store, None, "technical-concise")
    assert chosen.startswith("# Technical") and chosen != DEFAULT_STYLE


def test_paper_prompt_defers_voice_and_structure_to_writing_style():
    """The paper directive defers BOTH prose voice and narrative structure to the
    injected WRITING STYLE guide, keeping only task/correctness rules in the prompt."""
    from paperclaw.prompts.ideas import WRITE_PAPER_DIRECTIVE
    from paperclaw.prompts.writing_styles import DEFAULT_STYLE
    assert "{writing_style}" in WRITE_PAPER_DIRECTIVE
    # prose voice + narrative structure (funnel + section order) moved OUT of the
    # directive and INTO the style guide
    assert "avoid filler/AI tells" not in WRITE_PAPER_DIRECTIVE
    assert "FUNNEL → REVERSE FUNNEL" not in WRITE_PAPER_DIRECTIVE
    assert "SECTIONS, in this order" not in WRITE_PAPER_DIRECTIVE
    assert "funnel" in DEFAULT_STYLE.lower() and "reverse funnel" in DEFAULT_STYLE.lower()
    assert "Related Work" in DEFAULT_STYLE and "Preliminaries" in DEFAULT_STYLE
    # Method = end-to-end with an equation per step + architecture, but NOT experiment params
    flat = " ".join(DEFAULT_STYLE.split())
    assert "EQUATION for each step" in DEFAULT_STYLE
    assert "MODEL ARCHITECTURE" in DEFAULT_STYLE
    assert "do NOT enumerate experiment settings" in flat
    # correctness/task rules stay in the prompt
    assert "REAL measured numbers" in WRITE_PAPER_DIRECTIVE
    assert 'NEVER mention "hypothesis"' in WRITE_PAPER_DIRECTIVE


def test_rigor_rules_injected_into_both_paper_prompts():
    """Shared scientific-rigor rules are TASK/correctness (not writing style): they
    inject into BOTH paper system prompts via {rigor_rules}, and stay domain-neutral."""
    import re
    from paperclaw.prompts.ideas import WRITE_PAPER_DIRECTIVE
    from paperclaw.prompts.pipeline import LATEX_PAPER_SYSTEM, PAPER_RIGOR_RULES
    assert "{rigor_rules}" in WRITE_PAPER_DIRECTIVE and "{rigor_rules}" in LATEX_PAPER_SYSTEM
    for key in ("EVIDENCE-BOUNDING", "FAILURE-AWARE", "PER-REGIME",
                "CITE ORIGINAL PAPERS", "BASELINE MODERNITY",
                "FOCUS ON THE IDEA", "CITE PRIOR WORK INLINE", "DON'T DUMP CITATIONS"):
        assert key in PAPER_RIGOR_RULES
    # no field-specific (RL/ML) terms leaked in from the source prompt
    assert not re.search(r"reward|action space|episode|epoch|\bPPO\b|ResNet|BatchNorm|\bAdam\b",
                         PAPER_RIGOR_RULES, re.I)
