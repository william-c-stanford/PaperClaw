# Example papers

Complete, real runs that **PaperClaw produced end-to-end** — topic → domain → idea →
hypotheses → experiments → **compiled PDF** — each typeset with its **target venue's**
LaTeX template.

Every entry below is a **full idea workspace** (not just a PDF): the idea spec, the
hypothesis map, the per-hypothesis experiments with their results, the validated
bibliography, the venue template, and the LaTeX source alongside the compiled paper.

## Papers

### 📁 [Paper 1 — RC-Diff: Risk-Controlled Financial Diffusion with Path-Level Audits](<[Paper 1] rc-diff-risk-controlled-financial-diffusion/>)

Diffusion models for financial time series · Target Venue · 9 pp ·
➡️ **[read the compiled paper.pdf](<[Paper 1] rc-diff-risk-controlled-financial-diffusion/paper.pdf>)**

## What's in a paper directory

Each `[Paper N] …/` folder is a self-contained PaperClaw run:

| Path | What it is |
|---|---|
| `paper.pdf` | the final compiled manuscript |
| `paper.tex` | LaTeX source (target-venue template) |
| `IDEA.md` | the idea spec — background, gap, motivation, root hypotheses |
| `.hypothesis_map.json` | the hypothesis tree that was explored |
| `H1/`, `H2/`, `H3/` | per-hypothesis plan → experiment → report, with code, `results.json` & figures |
| `fig_*.png`, `make_fig_results.py` | paper figures and the script that drew them |
| `ref.bib` | bibliography, every entry validated against Crossref / OpenAlex |
| `venue/`, `*.sty` | the target-venue LaTeX template the paper was built on |

> Reproduce one yourself: `paperclaw auto run "<topic>"` — see the
> [project README](../../README.md#-two-ways-to-run-it).
