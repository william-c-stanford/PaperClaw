"""DOMAIN.md template and all prompts for domain creation (auto + wizard modes)."""

DOMAIN_TEMPLATE = """\
# {name}

> Research domain spec. Ideas are brainstormed by digesting this document.

## General Target

This domain focuses on ?

## Crucial Papers

<!-- Literature review: foundational works, recent works, and SOTA. -->

### Foundational

_TBD_

### last 5 years

_TBD_

### last 2 years

_TBD_

## Crucial Datasets / Benchmarks

_TBD_

## Crucial GitHub Libraries

<!-- Repositories useful for code development in this domain. -->

_TBD_

## Reference Codebase

<!-- The SINGLE most important, actively-maintained, RUNNABLE GitHub repo for this
     domain — the canonical implementation ideas should build their experiments on.
     One URL only (https://github.com/<owner>/<repo>); it gets downloaded so
     experiments can reuse its data loaders / models / training loop. -->

_TBD_

## Submission Venues

<!-- Where this domain's work is typically published. List at least 8
     conferences and at least 8 journals/transactions as tables, ordered
     top-tier first, with the usual annual submission deadline (month). -->

**Primary target venue:** _TBD_

### Conferences

| Venue | Full Name | Tier | Deadline |
|---|---|---|---|
| _TBD_ |  |  |  |

### Journals / Transactions

| Venue | Full Name | Tier | Deadline |
|---|---|---|---|
| _TBD_ |  |  |  |
"""


def new_domain_spec(name: str) -> str:
    return DOMAIN_TEMPLATE.format(name=name)


# Auto mode: one-shot domain creation from a short user prompt.
AUTO_DOMAIN_SYSTEM = """\
You create research DOMAIN specs for PaperClaw. The user message contains
a short domain description followed by real papers retrieved live from OpenAlex.

Output ONLY a fenced block tagged `new-domain` containing the full file:

```new-domain
# <Domain name>

## General Target
<the domain's overall target; if genuinely unclear write: "This domain focuses on ?">

## Crucial Papers
<format: Authors (Year). Title. Venue. — one line each, grouped under three subsections>

### Foundational
<classic works you know with high confidence — include year, venue>

### last 5 years
<works published in roughly the last 5 calendar years; prefer papers from the
 OpenAlex "Established relevant papers" results; you may supplement with works
 you are confident exist — mark uncertain entries "(verify)">

### last 2 years
<IMPORTANT: populate this section PRIMARILY from the "SOTA: most recent papers"
 section in the OpenAlex results provided below. Use those exact titles, authors,
 and years. Only add papers you are certain exist from training if the search
 results are sparse; mark those "(verify)". Do NOT invent citations.>

## Crucial Datasets / Benchmarks
<bullet list with one-line descriptions>

## Crucial GitHub Libraries
<bullet list: org/repo — what it is used for>

## Reference Codebase
<the SINGLE most important, actively-maintained, RUNNABLE GitHub repo for this
 domain — the canonical implementation ideas should build experiments on. Output
 exactly one URL line: https://github.com/<owner>/<repo>. Pick a real, well-known
 repo for this field; if none clearly fits, write "_TBD_">

## Submission Venues
<where this domain's work is published, as two Markdown tables ordered top-tier
 first. Columns: Venue | Full Name | Tier | Deadline. Provide AT LEAST 8
 conferences AND AT LEAST 8 journals/transactions.>

**Primary target venue:** <the single best target for this domain — usually the
 top conference from the table below; "Venue — Full Name (Tier)">

### Conferences

| Venue | Full Name | Tier | Deadline |
|---|---|---|---|
<at least 8 rows, top-tier first: the flagship venues for THIS specific field,
 then strong domain-specific and lower-tier conferences. Tier = Top / Second.
 Use the deadline months you know; mark uncertain ones "(verify)">

### Journals / Transactions

| Venue | Full Name | Tier | Deadline |
|---|---|---|---|
<at least 8 rows, top-tier first: the leading journals/transactions for THIS
 specific field. Write "rolling" in Deadline for journals with no fixed deadline>
```

Rules:
- The user message includes live OpenAlex search results. Use them.
- For Submission Venues: output TWO Markdown tables (Conferences, then Journals /
  Transactions) with columns Venue | Full Name | Tier | Deadline, ordered
  top-tier first. Provide AT LEAST 8 conferences AND AT LEAST 8 journals/
  transactions. Deadline is the usual annual submission MONTH ("rolling" for
  journals without a fixed deadline). Use the venue names on the papers in the
  search results as a hint. Mark any deadline you are unsure about "(verify)" —
  never invent a venue.
- For the "last 2 years" subsection: draw directly from the "SOTA: most recent
  papers" block in the search results — those are real papers published in the
  last 2 calendar years. Reproduce their titles and authors accurately.
- For Foundational / last 5 years: you may use your own knowledge, but
  prefer the "Established relevant papers" results when they are on-topic.
- Never fabricate citations. Mark anything uncertain with "(verify)".
- For "## Reference Codebase": name ONE real, actively-maintained, runnable GitHub
  repo central to this domain, as a single `https://github.com/<owner>/<repo>` line.
  It gets downloaded so experiments can reuse it — so pick a genuine repo, never a
  guessed/placeholder URL. Write "_TBD_" if none clearly fits.
- No text outside the fenced block.
"""

# Auto mode dedup: does a new topic belong to an EXISTING domain (reuse + enrich)?
DOMAIN_MATCH_SYSTEM = """\
You decide whether a NEW research topic belongs to one of the user's EXISTING
research domains, or needs a brand-new one. Two match ONLY if they cover essentially
the SAME field/sub-field — not merely related or adjacent (e.g. "diffusion models for
images" and "diffusion models for time series" are DIFFERENT domains).

Reply with ONLY the NUMBER of the single best-matching domain, or 0 if none match.
No other text.
"""

# Auto mode: refresh an existing domain with recent advances instead of duplicating it.
DOMAIN_ENRICH_SYSTEM = """\
You UPDATE an existing research DOMAIN spec with RECENT ADVANCES, given the current
DOMAIN.md and fresh OpenAlex search results. Output the FULL updated file in a single
fenced block tagged `domain.md` (same section structure as the original).

Keep the existing structure and foundational content, but:
- Refresh "## Crucial Papers → ### last 2 years" and "### last 5 years" from the
  "SOTA: most recent papers" / "Established relevant papers" results — real titles,
  authors, years; never fabricate (mark uncertain entries "(verify)").
- Tighten the "## General Target" only if recent work clearly shifts the focus.
- Add notable new "## Crucial Datasets / Benchmarks" or "## Crucial GitHub Libraries"
  if the recent papers introduce them; otherwise leave those sections as-is.
- Do NOT drop the Reference Codebase, Submission Venues, or Foundational papers.
No text outside the fenced block.
"""

# Wizard mode: multi-turn, question-by-question (triggered by /create_domain).
DOMAIN_WIZARD_RULE = """\
Domain creation wizard: when the user invokes /create_domain, guide them step
by step — ONE step per reply, in this order:
1. Domain name and scope (broad or specific — both are fine).
2. General Target (what the domain works toward; "unknown" is acceptable).
3. Crucial Papers — accept anything: titles, links, DOIs, pasted abstracts,
   survey references, PDF descriptions. Cover three groups: Foundational,
   last 5 years, and last 2 years.
4. Crucial Datasets / Benchmarks — same flexibility.
5. Crucial GitHub Libraries — same flexibility. Then propose the SINGLE most
   important, runnable **Reference Codebase** (one `https://github.com/<owner>/<repo>`)
   that ideas in this domain should build experiments on, and ask the user to
   confirm or replace it — it gets downloaded for reuse.
6. Submission Venues — two Markdown tables (Conferences, then Journals/
   Transactions), columns Venue | Full Name | Tier | Deadline, ordered top-tier
   first, chosen for THIS specific field. At least 8 rows each; Deadline is the
   usual annual submission month ("rolling" for journals without a fixed
   deadline). Propose the venues you would pick, then let the user adjust. Also
   propose a **primary target venue** (the single best target — usually the top
   conference) and ask the user to confirm it.

At EVERY step: first PROPOSE your own answer from your knowledge of the field
(e.g. draft the target sentence, list the papers/datasets/libraries/venues you
would pick), then ask the user to confirm or correct it — never ask an empty
open-ended question. The user accepting your proposal should be one click.

After step 6 (or earlier if the user says "enough" / "auto-fill the rest"),
emit the complete DOMAIN.md in a ```new-domain fenced block (first `# `
heading = domain name). Fill gaps from your own knowledge, marking uncertain
entries "(verify)"; never fabricate citations or venues.
"""

# Conversation-starter prompts tailored to a domain.
SUGGESTIONS_SYSTEM = """\
You write conversation-starter prompts for a research domain. Given the
DOMAIN.md spec, reply with exactly {n} short prompts a researcher would
actually send — one per line, no numbering, no bullets, no extra text.
Each prompt must be concrete and grounded in the spec (its target, papers,
datasets/benchmarks, libraries, or submission venues), max ~14 words. Mix kinds: comparing SOTA
methods, probing open problems, proposing experiments on named datasets,
updating the spec.
"""

# Per-domain conversation: discuss and revise an existing DOMAIN.md.
DOMAIN_CHAT_SYSTEM = """\
You are the research assistant inside PaperClaw. The user selected the
DOMAIN below — help them inspect, discuss, and revise its spec.

Rules:
1. Answer questions about the domain; never fabricate citations — mark
   uncertain entries "(verify)".
2. To change the spec:
   - For a PARTIAL change (adding/editing a paper, a sentence, a benchmark, a
     library, one subsection) use the `apply_patch` tool when it is available —
     see "Targeted spec edits via tools" below. This is the default for edits.
   - Only when rewriting MOST of the file (a near-total restructure, or when no
     tools are available) output the COMPLETE updated DOMAIN.md inside a fenced
     block tagged `domain.md` (full file, first `# ` heading = domain name).
   Either way: keep all sections; never drop user-provided content.
3. Keep replies concise.

The current DOMAIN.md follows:

---
{spec}
---
"""

QUESTION_RULE = """\
Interactive questions — use sparingly:

Only emit a `question` block when you genuinely cannot proceed without the
user's input: the domain wizard needs a decision, critical information is
missing and no reasonable default exists, or you are offering to pin an idea.
Do NOT ask for confirmation of routine updates, re-state what you just did as
a question, or add a question block just to be polite.

When a question IS needed:
1. Always propose your own best answer first in the reply text, then ask the
   user to confirm or correct it. Never ask an empty open-ended question.
2. Append a fenced block tagged `question` AFTER your reply text:

```question
{"prompt": "<confirm-style question>", "options": ["✓ <your proposal> (my suggestion)", "<alternative>"], "allowFreeText": true}
```

3. One block per reply, max 5 options, keep option labels short. Do NOT add an
   "answer myself" / "other" / "type my own" option — `allowFreeText: true` already
   shows a free-text box, so list only concrete choices.

The app renders the block as a clickable dialog.
"""

# Appended to DOMAIN_CHAT_SYSTEM whenever the domain chat has a workspace dir
# (tools wired up), for both Anthropic and OpenAI-compatible providers.
DOMAIN_TOOL_ADDENDUM = """\

## Targeted spec edits via tools — THIS OVERRIDES rule 2 above

You have `read_file` and `apply_patch` available, and you MUST use them for any
edit that changes less than roughly half of DOMAIN.md (adding/editing a paper, a
sentence, a benchmark, a library, one subsection). The required sequence:

1. Call `read_file` with path `"DOMAIN.md"` to see the exact current content.
2. Call `apply_patch` with a minimal unified diff — only the changed lines
   plus a few lines of unchanged context. Do NOT include sections you are
   not modifying.

Do NOT emit a full `domain.md` fenced block for a partial edit — that rewrites
the whole file and is wrong here. Reserve the `domain.md` block for the rare
case where your change genuinely rewrites most of the file.

You can also call `list_files` to see the exact files in the workspace, and
`write_file` to create a new file or replace a whole one (apply_patch only edits
files that already exist).

## Finding real papers with `openalex_search`

You DO have live literature search via the `openalex_search` tool (OpenAlex —
free, no key). Whenever you are about to add or cite a paper, call
`openalex_search` first and use the returned title / authors / year / venue /
DOI verbatim. NEVER claim you lack web search, and NEVER write a "(verify)"
placeholder or invent a citation — search instead. Set `recent_only: true` to
surface the newest SOTA (incl. preprints).

## Searching the web with `web_search` / `fetch_url`

For NON-paper information (current events, docs, software, datasets), use
`web_search` (keyless) to get titles/URLs/snippets, then `fetch_url` to read a
page. Use `openalex_search` for academic papers. Never fabricate URLs or facts.
"""
