"""Shim — prompts for domains now live in paperclaw.prompts.domains."""
from paperclaw.prompts.domains import *  # noqa: F401, F403
from paperclaw.prompts.domains import (  # noqa: F401
    AUTO_DOMAIN_SYSTEM,
    DOMAIN_CHAT_SYSTEM,
    DOMAIN_TEMPLATE,
    DOMAIN_WIZARD_RULE,
    QUESTION_RULE,
    SUGGESTIONS_SYSTEM,
    new_domain_spec,
)
