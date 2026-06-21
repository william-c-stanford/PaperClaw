"""Built-in prose-style guides for the paper-writing step.

These are the seed content for the writing-style library (`writing_styles.py`,
which owns the disk I/O). Each guide is markdown injected verbatim into
`WRITE_PAPER_DIRECTIVE`'s `{writing_style}` placeholder, so it governs HOW the
prose reads (tone / structure / phrasing) — SEPARATE from venue formatting
(`venue/STYLE.md` / `.sty`), which governs layout.

The guides are DOMAIN-NEUTRAL: they describe style, structure, and rigor that
apply to any field (use neutral placeholders like "the primary metric" rather
than naming a specific dataset, model, or venue). Detailed standards are distilled
from analyses of accepted papers at leading venues (ported from the ResearchClaw
sibling project's writing prompts). The first `# heading` of each guide is used as
its display title.

`DEFAULT_STYLE` is the house prose voice applied whenever the user has NOT picked a
named style (the dropdown's "Default"). It holds the baseline "write well" rules
that used to be hardcoded in the paper-writing system prompt — keeping prose-style
guidance HERE, not in the prompt, so the prompt carries only structure/correctness
rules. A selected `BUILTIN_STYLES` entry (or a user style) REPLACES it.
"""

# Applied when no named style is chosen — the baseline prose voice for every paper.
DEFAULT_STYLE = """\
# Default house style

The baseline voice AND narrative structure for a clear, credible research paper —
readable, precise, and free of AI tells. THE SINGLE GOAL is to CONVEY THE IDEA
clearly to the reader; the structure below is deliberate, but it exists only to
serve that — every section must make the idea easier to grasp; cut anything that
doesn't.

## Voice
- Lead with the concrete problem and the concrete result; do not bury the
  contribution under a history-of-the-field windup.
- Active voice, one idea per sentence; vary sentence length so it never reads as a
  monotone. Don't start three consecutive sentences with "We" / "The" / "Our".
- Quantify instead of praising: "a 3.1% improvement on the primary metric", not
  "significantly better". Use precise technical terms over hedging.
- Avoid filler and AI tells: "In this paper we propose a novel…", "extensive
  experiments demonstrate…", "plays a crucial role", "it is worth noting that".
- Be honest about what did and did not work; a clean negative or a modest-but-robust
  gain reads as credible, not weak. Consolidate caveats — state each ONCE.

## Narrative structure — a funnel → reverse funnel
Order the paper: Title, Abstract, Introduction, Related Work, Preliminaries, Method,
Experiments, Conclusion, References, then OPTIONALLY an Appendix (`\\appendix`, after
the bibliography) for proofs, derivations, and extra tables/figures.
- INTRODUCTION = funnel (BROAD → narrow): open on the field/application and why it
  matters, then the relevant prior work, then the specific open problem/gap,
  progressively narrowing to YOUR approach and contributions at the end. CITE DENSELY:
  in the Introduction nearly EVERY sentence — from the broad application/domain framing
  all the way through the prior work — carries an inline citation to a specific paper;
  the ONLY uncited sentences are the ones stating YOUR approach, contributions, and
  findings. The contributions are the IDEA and the FINDINGS, not a to-do list.
- PRELIMINARIES = the notation, problem formulation, and background the reader needs
  to follow the Method.
- METHOD = the focused core: present the COMPLETE method end-to-end — from the input /
  data notation, through every transformation, to the final output — AND discuss the
  MODEL ARCHITECTURE. Walk the pipeline step by step and give an EQUATION for each step
  (define every symbol — sets, indices, dimensions, operators — where it first appears);
  the prose explains the intuition around the equations, it does not replace them. This
  holds EVEN WHEN you reuse or build on an existing method: restate its formulation in
  YOUR notation with equations (a citation is NOT a substitute for the math), then make
  explicit what you changed and why. Keep the Method about the METHOD: do NOT enumerate
  experiment settings or hyperparameters here (they belong in the Experiments setup) —
  mention an experimental detail only when it is essential to understanding the method
  itself.
- EXPERIMENTS = evidence that examines the idea from several angles: a setup
  subsection (datasets / baselines / metrics), then the MAIN RESULTS, then KEY
  ANALYSES — ablations that isolate each component, plus the relevant of
  hyperparameter / sensitivity analysis and a case study.
- CONCLUSION = reverse funnel (narrow → BROAD): start from your findings, argue
  OUTWARD across the method's facets (what each result implies, when it helps,
  generalization, limitations), widening to broader impact, then a broad takeaway.
"""

BUILTIN_STYLES: dict[str, str] = {
    "technical-concise": """\
# Technical / Concise

A terse, results-first style for research papers at leading venues. Every sentence
earns its place; the contribution is unmistakable by the end of the first
paragraph.

## Voice
- Lead with the concrete problem and the concrete result. State the contribution
  in the FIRST paragraph — no throat-clearing, no history-of-the-field windup.
- Short, declarative sentences in the active voice ("We measure…", "The method
  predicts…"). One idea per sentence; vary length so it never reads as a monotone.
- Confident but precise. Prefer exact technical terms over hedging; let the numbers
  do the persuading.

## Quantify, don't praise
- Replace vague praise with a measured claim: "a 3.1% improvement on the primary
  metric", not "significantly better"; "1.4× faster at matched quality", not "much
  faster".
- A comparison claim ("A outperforms B") needs a number AND, where you have the
  runs, a significance signal (CI or p-value). If it isn't significant, say
  "comparable" / "competitive" — never imply superiority you can't back.

## Notation & equations
- Define notation once and reuse it consistently; never reuse a symbol for two
  meanings. Equations carry the argument; the prose explains the intuition around
  them, not the LaTeX.

## Ban list (these read as AI/filler — do not use)
- "In this paper we propose a novel…", "extensive experiments demonstrate…",
  "plays a crucial role", "it is worth noting that", "to the best of our
  knowledge" (unless literally true and load-bearing).
- Don't start three consecutive sentences with "We" / "The" / "Our".

## Honesty
- Report limitations and negative results plainly — once, in Limitations. A
  clean negative or a modest-but-robust gain reads as credible, not weak.
""",
    "narrative": """\
# Narrative / Motivation-driven

A story-driven style that carries a broad reader from a real problem to your
method without losing rigor. The paper reads as a cohesive argument, not a
technical report or a bullet dump.

## The funnel
- Open on a real-world tension or a concrete failure of current methods, then
  narrow to the specific gap — a funnel the reader follows without the jargon.
- Motivate every technical choice BEFORE introducing it: say WHY, then HOW.
- Anchor abstract ideas to a recurring running example or dataset.

## Paragraph discipline (every body paragraph)
1. TOPIC SENTENCE — the claim or finding this paragraph makes.
2. EVIDENCE — the data, citation, or reasoning that supports it.
3. ANALYSIS — what the evidence means and why it matters.
4. TRANSITION — a sentence that hands off to the next paragraph's topic
   ("Building on this…", "In contrast to prior work…").

## Forbidden patterns
- Bullet/numbered lists in the body (allowed ONLY in the Introduction's
  contributions list and the Limitations section).
- Opening a paragraph with "Table X shows…" before any context.
- Repeating the same sentence structure three+ times in a row.

## BAD vs GOOD method description
BAD (bullet-list style):
  "Our method has three components: - Component A - Component B - Component C"
GOOD (narrative):
  "Our method builds on the insight that <core problem> stems from <root cause>.
   To address this we introduce <Name>, a three-stage framework. First, <Stage 1>
   maps inputs to <representation>; these feed <Stage 2>, enabling <benefit>
   without <drawback of prior work>. Crucially, <Stage 3> (cite the original
   technique) triggers <mechanism> when <condition> holds."
Replace every <placeholder> with your real details — never paste the template.

## Still rigorous
- Every claim is backed by a number or a citation; the story serves the evidence,
  not the reverse. Occasional first-person framing of decisions ("We first tried
  X, but it…") is welcome when it makes the reasoning legible — kept disciplined,
  never chatty.
""",
    "formal-theoretical": """\
# Formal / Theoretical

A precise, measured style for method papers with theory or careful analysis.
Persuasion comes from exactness, not flourish.

## Setup before results
- State assumptions, definitions, and the problem setup explicitly before any
  result. Use `\\begin{definition}` / `\\begin{theorem}` / `\\begin{proposition}`
  /`\\begin{lemma}` for the key claims; number them and refer back by number.
- Begin the Method with a problem formulation: the notation, the objects, and the
  objective or criterion being analyzed, in that order.

## Claim exactly as strong as the evidence
- Bound every claim ("under Assumption A1, …"); never overclaim. Explicitly
  distinguish what is PROVEN from what is EMPIRICAL.
- If a result is conditional or holds only in a regime, say so in the statement
  itself, not in a later caveat.

## Notation
- One consistent, carefully chosen notation. Introduce each symbol where it first
  appears; never overload a symbol with two meanings. Prefer a short notation
  table when the symbol count is high.

## Proofs
- Give a proof or proof sketch for each theoretical claim. Defer long proofs to an
  appendix, but state the KEY IDEA (one or two sentences) in the main text so the
  argument is followable without flipping to the appendix.
- After a theorem, add one plain-language sentence on what it means and why it
  matters — formal but readable.

## Tone
- Measured and impersonal where it aids precision; avoid rhetorical emphasis,
  exclamation, and superlatives. Let the bounds and the constants speak.
""",
    "top-venue": """\
# Top-Venue / Award-paper standards

A comprehensive house style distilled from accepted papers at leading venues. Use
this when you want the draft to hit the structural and citation bar of a strong
submission, not just read cleanly. Combine its STRUCTURE rules with any of the
voice guides above.

## Title (8-14 words, never > 18)
- Preferred: `SystemName: Descriptive Subtitle` (give the method a memorable,
  pronounceable name). Alternative: a declarative statement that surprises.
- The title may ONLY claim what the experiments actually measured. If coverage is
  partial, prefer "Toward…" / "An Empirical Study of…" over a global "X Beats Y".
- Banned: "A Novel Approach to…", "Investigating…", "Exploring…".

## Abstract (180-220 words, problem→method→results)
- 1-2 sentences: the problem / gap (a status-quo critique).
- 1-2 sentences: name your method and its key insight.
- 2-3 concrete quantitative claims — at least one RELATIVE ("a 36.7% improvement
  over the baseline") and one ABSOLUTE (a headline score on a named benchmark). No
  per-run ranges, no defensive hedging.

## Section budget & citation targets
- INTRODUCTION (~4 paragraphs): motivation → gap (cite 3-5) → approach →
  contributions (a 3-4 item bullet list, the only list allowed here). Cite 8-12
  works total.
- RELATED WORK: organized into 2-3 sub-topics, NOT a flat list; end each with how
  YOUR work differs. Aim for >=15 unique references in this section.
- METHOD: problem formulation (notation, objective) first; pseudocode in an
  `algorithm` environment; flowing prose, not bullets.
- EXPERIMENTS: a setup subsection (datasets, baselines, metrics, conditions) + a
  settings/configuration table (Table 1); reference every figure in prose ("As
  shown in Figure 1…") and CITE each baseline's paper, don't just name it.
- RESULTS: a main-results table + an ablation table with descriptive captions;
  analysis paragraphs that connect numbers to insight — do not restate the same
  numbers already in Experiments.
- DISCUSSION / LIMITATIONS (200-300 words, 3-5 concrete items) / CONCLUSION
  (2-3 sentences + 2-3 of future work). Citations belong in EVERY section.
- Across the paper cite 25-40 unique, directly relevant references, and always
  cite the ORIGINAL paper that introduced any established technique or tool you
  use, never a survey or follow-up in its place.

## Anti-repetition
- Each specific number (e.g. "0.7667", "36.7%") appears in at most TWO places —
  once where first reported (Results/Experiments) and optionally once in the
  Abstract. Introduction, Discussion, and Conclusion refer to results
  QUALITATIVELY ("substantially outperformed") without repeating exact figures.

## Anti-hedging
- Consolidate ALL caveats in Limitations, stated once. Banned from the body:
  "we do not claim", "we cannot prove", "only N runs", "we intentionally frame
  this conservatively". Reframe confidence positively ("our results provide
  evidence for X").
- Turn a negative result into an INSIGHT: not "our method failed to beat the
  baseline", but "surprisingly, the standard baseline proved competitive,
  suggesting <why> — with practical implications for <where>".

## Evidence-bounding (a violation reads as a reject)
- Every claim in the title, abstract, and conclusion is directly backed by a
  specific metric in the results. Never fabricate or round-up numbers; report
  per-regime results rather than only aggregates when you claim robustness.
""",
}
