import { useEffect, useMemo, useState, useRef, KeyboardEvent, ChangeEvent } from 'react'
import { api } from '../../api'
import type { Skill } from '../../types'
import s from './styles.module.css'

interface Props { onSend: (text: string) => void; ideaActive?: boolean }

const SUGGESTIONS = [
  '/create_domain',
  'Survey diffusion models for time series',
  'Find seminal papers on neural ODEs',
  'Generate a research hypothesis'
]

const HISTORY_KEY = 'paperclaw_chat_history'
const HISTORY_MAX = 100

export default function InputArea({ onSend, ideaActive }: Props) {
  const [text, setText] = useState('')
  const [skills, setSkills] = useState<Skill[]>([])
  const [menuIndex, setMenuIndex] = useState(0)
  const ref = useRef<HTMLTextAreaElement>(null)
  // Sent-message history (browsed with ↑/↓), persisted across reloads.
  const [history, setHistory] = useState<string[]>([])
  const histIdx = useRef<number | null>(null)  // null = editing a fresh draft
  const draftRef = useRef('')                  // the draft stashed when nav starts
  const caretEnd = useRef(false)               // move caret to end after a nav setText
  const sentGuard = useRef<string | null>(null)  // blocks an immediate duplicate send

  useEffect(() => {
    api.getSkills().then(setSkills).catch(() => {})
    try {
      const h = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
      if (Array.isArray(h)) setHistory(h.filter(x => typeof x === 'string'))
    } catch { /* ignore */ }
  }, [])

  // Auto-resize, and (after a history nav) drop the caret at the end.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
    if (caretEnd.current) { el.selectionStart = el.selectionEnd = el.value.length; caretEnd.current = false }
  }, [text])

  // Slash-command menu: open while the first token starts with "/"
  const menu = useMemo<Skill[]>(() => {
    if (!text.startsWith('/')) return []
    const token = text.split(/\s/, 1)[0].toLowerCase()
    if (text.includes(' ') && skills.some(sk => sk.command === token)) return []
    // hide idea-only skills (write paper, validate refs, hypothesis steps…) until an
    // idea is selected — they don't do anything useful otherwise.
    return skills.filter(sk =>
      sk.command.toLowerCase().startsWith(token) && (ideaActive || !sk.requiresIdea))
  }, [text, skills, ideaActive])

  useEffect(() => { setMenuIndex(0) }, [menu.length])

  const pick = (skill: Skill) => {
    setText(`${skill.command} `)
    ref.current?.focus()
  }

  const send = () => {
    const val = text.trim()
    // Guard against a double-fire (key-repeat / a stray second Enter before the input
    // clears) sending the same message twice. setText('') is async, so a ref lock is
    // the reliable check.
    if (!val || sentGuard.current === val) return
    sentGuard.current = val
    setTimeout(() => { if (sentGuard.current === val) sentGuard.current = null }, 500)
    onSend(val)
    setHistory(h => {
      const next = (h[h.length - 1] === val ? h : [...h, val]).slice(-HISTORY_MAX)
      try { localStorage.setItem(HISTORY_KEY, JSON.stringify(next)) } catch { /* ignore */ }
      return next
    })
    histIdx.current = null
    draftRef.current = ''
    setText('')
  }

  // Browse sent-message history with ↑/↓ — but only at the text boundaries, so
  // arrows still move the caret within a multi-line draft (like a shell / Claude).
  const navHistory = (e: KeyboardEvent<HTMLTextAreaElement>): boolean => {
    if (!history.length) return false
    const el = e.currentTarget
    const v = el.value
    const pos = el.selectionStart ?? 0
    if (e.key === 'ArrowUp') {
      const firstNL = v.indexOf('\n')
      if (firstNL !== -1 && pos > firstNL) return false  // not on the first line
      e.preventDefault()
      if (histIdx.current === null) { draftRef.current = text; histIdx.current = history.length }
      histIdx.current = Math.max(0, histIdx.current - 1)
      caretEnd.current = true
      setText(history[histIdx.current])
      return true
    }
    if (e.key === 'ArrowDown') {
      if (histIdx.current === null) return false  // not browsing history
      const lastNL = v.lastIndexOf('\n')
      if (pos <= lastNL) return false  // not on the last line
      e.preventDefault()
      if (histIdx.current < history.length - 1) {
        histIdx.current += 1
        setText(history[histIdx.current])
      } else {
        histIdx.current = null
        setText(draftRef.current)
      }
      caretEnd.current = true
      return true
    }
    return false
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (menu.length > 0) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setMenuIndex(i => (i + 1) % menu.length); return }
      if (e.key === 'ArrowUp') { e.preventDefault(); setMenuIndex(i => (i - 1 + menu.length) % menu.length); return }
      if (e.key === 'Tab' || (e.key === 'Enter' && text.trim() !== menu[menuIndex].command)) {
        e.preventDefault(); pick(menu[menuIndex]); return
      }
      if (e.key === 'Escape') { setText(text.replace(/^\//, '')); return }
    }
    if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && navHistory(e)) return
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const onInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
    histIdx.current = null  // user edited → leave history browsing
    setText(e.target.value)
  }

  return (
    <div className={s.inputArea}>
      <div className={s.inputWrapOuter}>
        {menu.length > 0 && (
          <div className={s.cmdMenu}>
            {menu.map((sk, i) => (
              <button
                key={sk.command}
                className={`${s.cmdItem} ${i === menuIndex ? s.cmdItemActive : ''}`}
                onMouseEnter={() => setMenuIndex(i)}
                onClick={() => pick(sk)}
              >
                <span className={s.cmdName}>{sk.command}</span>
                <span className={s.cmdDesc}>{sk.description}</span>
              </button>
            ))}
          </div>
        )}
        <div className={s.inputWrap}>
          <textarea
            ref={ref}
            className={s.inputBox}
            value={text}
            onChange={onInput}
            onKeyDown={onKey}
            placeholder="Ask anything… type / for commands"
            rows={1}
          />
          <button className={s.sendBtn} disabled={!text.trim()} onClick={send} title="Send (Enter)">
            ↑
          </button>
        </div>
      </div>
      <p className={s.inputHint}>
        <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> new line · <kbd>↑</kbd><kbd>↓</kbd> history · <kbd>/</kbd> commands
      </p>
    </div>
  )
}

export { SUGGESTIONS }
