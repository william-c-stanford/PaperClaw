import { useEffect, useState } from 'react'
import { api } from '../../api'
import type { ChatContext } from '../../types'
import s from './styles.module.css'

interface Props {
  onPick: (ctx: ChatContext) => void
  isStreaming?: boolean
}

const KIND_ICON: Record<ChatContext['kind'], string> = {
  scratch: '💬',
  domain: '🌐',
  seed: '📝',
  idea: '💡'
}

function ago(ts: number): string {
  const sec = Date.now() / 1000 - ts
  if (sec < 60) return 'just now'
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  return `${Math.floor(sec / 86400)}d ago`
}

export default function HistoryMenu({ onPick, isStreaming }: Props) {
  const [open, setOpen] = useState(false)
  const [contexts, setContexts] = useState<ChatContext[]>([])

  useEffect(() => {
    if (!open) return
    api.getContexts().then(setContexts).catch(() => setContexts([]))
    const close = () => setOpen(false)
    window.addEventListener('click', close)
    return () => window.removeEventListener('click', close)
  }, [open])

  return (
    <div className={s.historyWrap} onClick={e => e.stopPropagation()}>
      <button
        className={s.historyBtn}
        onClick={() => setOpen(v => !v)}
        title="All conversations"
      >
        {isStreaming ? <span className={s.chatSpinner} /> : '🕘'} History
      </button>

      {open && (
        <div className={s.historyMenu}>
          {contexts.length === 0 ? (
            <p className={s.historyEmpty}>No conversations yet</p>
          ) : (
            contexts.map(ctx => (
              <button
                key={ctx.contextId}
                className={s.historyItem}
                onClick={() => { onPick(ctx); setOpen(false) }}
              >
                <span className={s.historyIcon}>{KIND_ICON[ctx.kind]}</span>
                <span className={s.historyTitle}>{ctx.title}</span>
                <span className={s.historyMeta}>
                  {ctx.messageCount} · {ago(ctx.lastTimestamp)}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}
