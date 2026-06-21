"""Prompt library for PaperClaw — all system prompts in one place.

Sub-modules:
  prompts.ideas    — IDEA.md template, chat/brainstorm prompts
  prompts.domains  — DOMAIN.md template, auto/wizard/chat prompts
  prompts.pipeline — 4-phase autonomous research pipeline prompts
  prompts.hardware — LLM hardware-assessment prompt
  prompts.writing_styles — built-in prose-style guides for paper writing
"""

from paperclaw.prompts.domains import (
    AUTO_DOMAIN_SYSTEM,
    DOMAIN_CHAT_SYSTEM,
    DOMAIN_TEMPLATE,
    DOMAIN_TOOL_ADDENDUM,
    DOMAIN_WIZARD_RULE,
    QUESTION_RULE,
    SUGGESTIONS_SYSTEM,
    new_domain_spec,
)
from paperclaw.prompts.ideas import (
    BRAINSTORM_DRAFT_SYSTEM,
    BRAINSTORM_SYSTEM,
    CHAT_SYSTEM,
    CHAT_TOOL_ADDENDUM,
    IDEA_GENERATION_DIRECTIVE,
    IDEA_TEMPLATE,
    NEW_IDEA_RULE,
    SCRATCH_SYSTEM,
    SEED_CHAT_SYSTEM,
    new_spec,
)
from paperclaw.prompts.hardware import HARDWARE_ASSESS_SYSTEM
from paperclaw.prompts.pipeline import (
    ANALYSIS_SYSTEM,
    EXPERIMENT_SYSTEM,
    PAPER_TEMPLATE,
    PLAN_SYSTEM,
)
from paperclaw.prompts.writing_styles import BUILTIN_STYLES, DEFAULT_STYLE

__all__ = [
    # ideas
    "IDEA_TEMPLATE", "new_spec", "CHAT_SYSTEM", "CHAT_TOOL_ADDENDUM",
    "NEW_IDEA_RULE", "SCRATCH_SYSTEM", "IDEA_GENERATION_DIRECTIVE",
    "BRAINSTORM_SYSTEM", "BRAINSTORM_DRAFT_SYSTEM", "SEED_CHAT_SYSTEM",
    # domains
    "DOMAIN_TEMPLATE", "new_domain_spec", "AUTO_DOMAIN_SYSTEM",
    "DOMAIN_WIZARD_RULE", "SUGGESTIONS_SYSTEM", "DOMAIN_CHAT_SYSTEM",
    "DOMAIN_TOOL_ADDENDUM", "QUESTION_RULE",
    # pipeline
    "PLAN_SYSTEM", "EXPERIMENT_SYSTEM", "ANALYSIS_SYSTEM", "PAPER_TEMPLATE",
    # hardware
    "HARDWARE_ASSESS_SYSTEM",
    # writing styles
    "BUILTIN_STYLES", "DEFAULT_STYLE",
]
