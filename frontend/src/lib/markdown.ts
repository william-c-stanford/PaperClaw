import { marked } from 'marked'
import DOMPurify from 'dompurify'
import katex from 'katex'

// LaTeX math segments are extracted and rendered with KaTeX BEFORE markdown
// parsing, so marked never mangles `_`, `^`, `\` inside formulas. Rendered
// (trusted) KaTeX HTML is re-inserted AFTER sanitization so DOMPurify doesn't
// strip KaTeX's markup. Supports both $…$/$$…$$ and \(…\)/\[…\] delimiters.

interface MathToken { placeholder: string; html: string }

function renderMath(tex: string, display: boolean): string {
  try {
    return katex.renderToString(tex.trim(), {
      displayMode: display,
      throwOnError: false,
      output: 'html',
    })
  } catch {
    return display ? `<pre>${tex}</pre>` : `<code>${tex}</code>`
  }
}

function extractMath(src: string): { text: string; tokens: MathToken[] } {
  const tokens: MathToken[] = []
  let i = 0
  const stash = (tex: string, display: boolean): string => {
    const placeholder = `@@MATH_${i++}@@`
    tokens.push({ placeholder, html: renderMath(tex, display) })
    return placeholder
  }

  let text = src
  // Display first (longer delimiters), then inline.
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_m, tex) => stash(tex, true))
  text = text.replace(/\\\[([\s\S]+?)\\\]/g, (_m, tex) => stash(tex, true))
  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (_m, tex) => stash(tex, false))
  // Inline $…$ — no newline, not an escaped \$, avoids matching empty $$.
  text = text.replace(/(?<!\\)\$(?!\s)([^\n$]+?)(?<!\\)\$/g, (_m, tex) => stash(tex, false))
  return { text, tokens }
}

// `breaks: true` makes every single newline a <br>. Specs are line-oriented
// (e.g. DOMAIN.md lists one paper per line without blank lines between them);
// without this they collapse into one wrapped paragraph instead of rows.
const MARKED_OPTS = { async: false, breaks: true } as const

/** Parse Markdown with embedded LaTeX math into sanitized HTML. */
export function renderMarkdownWithMath(src: string): string {
  const { text, tokens } = extractMath(src)
  const rawHtml = marked.parse(text, MARKED_OPTS) as string
  let safe = DOMPurify.sanitize(rawHtml)
  for (const { placeholder, html } of tokens) {
    safe = safe.replace(placeholder, html)
  }
  return safe
}

/** Plain Markdown → sanitized HTML (no math). */
export function renderMarkdown(src: string): string {
  return DOMPurify.sanitize(marked.parse(src, MARKED_OPTS) as string)
}

/** Rewrite RELATIVE <img> srcs (e.g. `fig1.png`, or a stale `experiments/fig1.png`)
 *  to a fetchable workspace URL so inline figures load. Resolved by BASENAME (robust
 *  to wrong prefixes); absolute/data/blob srcs are left alone. */
function rewriteImgs(html: string, fileUrl: (name: string) => string): string {
  return html.replace(/(<img\b[^>]*?\bsrc=")([^"]+)(")/gi, (full, pre, url, post) => {
    if (/^(https?:|data:|blob:|\/)/i.test(url)) return full
    const name = url.split('/').pop() || url
    return pre + fileUrl(name) + post
  })
}

/** `renderMarkdown` + workspace figure resolution. */
export function renderMarkdownWithFigures(src: string, fileUrl: (name: string) => string): string {
  return rewriteImgs(renderMarkdown(src), fileUrl)
}

/** `renderMarkdownWithMath` + workspace figure resolution (used by the paper view). */
export function renderMarkdownWithMathAndFigures(src: string, fileUrl: (name: string) => string): string {
  return rewriteImgs(renderMarkdownWithMath(src), fileUrl)
}
