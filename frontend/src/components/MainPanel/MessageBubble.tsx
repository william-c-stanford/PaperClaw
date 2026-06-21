import { useMemo } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import type { Message } from '../../types'
import s from './styles.module.css'

function formatTime(ts: number) {
  // Backend timestamps are epoch seconds; local optimistic ones already match.
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const md = (text: string) => DOMPurify.sanitize(marked.parse(text, { async: false }) as string)

interface Props { message: Message }

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const isStreaming = message.status === 'streaming'
  const parts = message.parts

  // Final answer markdown (only when there's no part timeline — e.g. messages
  // reloaded from the server, which carry plain content but no parts/tools).
  const html = useMemo(() => {
    if (isUser || isStreaming || parts || !message.content) return ''
    return md(message.content)
  }, [isUser, isStreaming, parts, message.content])

  // Chronological timeline: narration (markdown) + tool calls interleaved.
  const timeline = !isUser && parts && parts.length ? (
    <div className={s.bubble}>
      {parts.map((p, i) => p.kind === 'text'
        ? <div key={i} className={s.bubbleMarkdown} dangerouslySetInnerHTML={{ __html: md(p.text) }} />
        : (
          <div key={i} className={s.toolRow}>
            <span className={s.toolIcon}>🔧</span>
            <code className={s.toolName}>{p.name}</code>
            {p.arg && <span className={s.toolArg}>{p.arg}</span>}
            {p.detail && <span className={s.toolDetail}>{p.detail}</span>}
          </div>
        ))}
      {isStreaming && <span className={s.cursor} />}
      {!isStreaming && message.specUpdated && (
        <div><span className={s.specBadge}>📋 spec updated</span></div>
      )}
    </div>
  ) : null

  return (
    <div className={`${s.messageGroup} ${isUser ? s.user : s.assistant}`}>
      <div className={s.messageMeta}>
        <span className={s.messageRole}>{isUser ? 'You' : 'PaperClaw'}</span>
        <span className={s.messageTime}>{formatTime(message.timestamp)}</span>
        {message.servedModel && (
          <span className={s.messageModel} title="Model that served this reply (from the API response)">
            {message.servedModel}
          </span>
        )}
      </div>

      {!isUser && message.thinking && (
        <details className={s.thinkingBox} open={isStreaming}>
          <summary className={s.thinkingSummary}>🧠 Thinking</summary>
          <div className={s.thinkingText}>{message.thinking}</div>
        </details>
      )}

      {!isUser && message.todos && message.todos.length > 0 && (() => {
        const done = message.todos.filter(t => t.status === 'completed').length
        return (
          <div className={s.todoBox}>
            <div className={s.todoHead}>📋 Plan · {done}/{message.todos.length}</div>
            {message.todos.map((t, i) => (
              <div key={i} className={`${s.todoItem} ${s['todo_' + t.status]}`}>
                <span className={s.todoMark}>
                  {t.status === 'completed' ? '☑' : t.status === 'in_progress' ? '◔' : '☐'}
                </span>
                <span className={s.todoText}>{t.content}</span>
              </div>
            ))}
          </div>
        )
      })()}

      {timeline ? (
        timeline
      ) : isStreaming ? (
        <div className={`${s.bubble} ${s.streaming}`}>
          <span className={s.dot} />
          <span className={s.dot} />
          <span className={s.dot} />
          {message.statusMessage && <span className={s.streamStatus}>{message.statusMessage}</span>}
        </div>
      ) : isUser ? (
        <div className={s.bubble}>{message.content}</div>
      ) : (html || message.specUpdated) ? (
        <div className={s.bubble}>
          <div className={s.bubbleMarkdown} dangerouslySetInnerHTML={{ __html: html }} />
          {message.specUpdated && (
            <div><span className={s.specBadge}>📋 spec updated</span></div>
          )}
        </div>
      ) : (message.thinking || (parts && parts.length)) ? null : (
        // a finished reply with no text/timeline/thinking — show a subtle note
        // instead of an empty bubble (e.g. the agent only wrote files via tools)
        <div className={`${s.bubble} ${s.muted ?? ''}`}>✓ Done — see the panel on the right.</div>
      )}
    </div>
  )
}
