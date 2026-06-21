"""Autonomous agents for PaperClaw.

Houses experiment runners that drive an LLM to carry out a hypothesis's
experiment:

  * `run_agentic_experiment` (`coding_agent.py`) — our own read→write→run→
    inspect→fix loop over ``llm.stream_chat_thinking`` (no external deps).
  * `run_cli_agent` (`cli_agent.py`) — shells out to an external **headless**
    coding-agent CLI (claude / opencode / openhands) and streams its stdout.

New agents (e.g. a literature agent, a review agent) go here.
"""

from paperclaw.agents.cli_agent import agent_command_available, run_cli_agent
from paperclaw.agents.coding_agent import run_agentic_experiment

__all__ = ["run_agentic_experiment", "run_cli_agent", "agent_command_available"]
