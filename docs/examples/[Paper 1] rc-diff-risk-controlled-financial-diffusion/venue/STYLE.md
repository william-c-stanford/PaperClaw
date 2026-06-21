# AAAI-26 Venue Style

## Sources
- Official AAAI-26 Submission Instructions: https://aaai.org/conference/aaai/aaai-26/submission-instructions/
- AAAI-26 Author Kit download: https://aaai.org/authorkit26-1/
- AAAI 2026 Overleaf mirror of author-kit source: https://www.overleaf.com/latex/templates/aaai-2026-press-formatting-instructions-for-authors-using-latex/qnpmwrzmddjj

## Venue + Year
AAAI-26, the 40th Annual AAAI Conference on Artificial Intelligence, January 20--27, 2026, Singapore.

## Downloaded Template
- Local file: `venue/authorkit26.zip`
- Downloaded from the official AAAI author-kit URL above.
- The visible LaTeX source uses `aaai2026.sty` via `\usepackage[submission]{aaai2026}`.
- The visible anonymous submission template is `anonymous-submission-latex-2026.tex`.

## Page Limit
- Review submissions may include up to 7 pages of technical content.
- Additional pages are allowed solely for references and the reproducibility checklist.
- Acknowledgements must be omitted for review submissions.

## Paper Size / Columns
- US Letter paper: 8.5 in x 11 in.
- AAAI two-column, camera-ready style.
- Submit a trouble-free, high-resolution PDF using Type 1 or TrueType fonts.

## Font + Size / Required Style Preamble
The AAAI-26 LaTeX source shows:
- `\documentclass[letterpaper]{article}`
- `\usepackage[submission]{aaai2026}`
- Required packages include `times`, `helvet`, `courier`, `url` with `hyphens`, `graphicx`, `natbib`, and `caption`.
- `\frenchspacing`, `\pdfpagewidth=8.5in`, and `\pdfpageheight=11in` are marked as required in the template source.

## Required Sections / Submission Content
- Abstract and full paper are required submissions through OpenReview.
- Review PDF should be anonymous and omit author/affiliation information and acknowledgements.
- References and reproducibility checklist may appear after the technical-content page limit.

## Key Rules
- Double-blind review: remove author and affiliation information from the review submission.
- Use AAAI formatting; submissions must conform to the AAAI-26 author kit.
- Reference style is `natbib`-based in the provided LaTeX template.
- Do not alter margins, font sizes, or spacing to bypass limits.
- The template lists disallowed packages including `authblk`, `balance`, `CJK`, `float`, `flushend`, `fontenc`, `fullpage`, `geometry`, `hyperref`, `multicol`, `setspace`, `titlesec`, `ulem`, and `wrapfig`.
- The template lists disallowed commands including `\nocopyright`, `\addtolength`, `\balance`, `\baselinestretch`, `\clearpage`, `\newpage`, `\pagebreak`, `\pagestyle`, and `\tiny`.

