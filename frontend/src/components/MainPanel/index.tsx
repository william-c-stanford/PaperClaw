import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'
import InputArea, { SUGGESTIONS } from './InputArea'
import QuestionPanel from './QuestionPanel'
import HistoryMenu from './HistoryMenu'
import ExperimentMonitor from './ExperimentMonitor'
import AutoRunBanner from './AutoRunBanner'
import type { AutoRunStatus, ChatContext, Domain, ExperimentJob, Idea, Message, Seed } from '../../types'
import s from './styles.module.css'

/** A message typed during a streaming reply, waiting to be auto-sent. */
interface QueuedMessage { id: string; text: string }

interface Props {
  messages: Message[]
  activeIdea?: Idea
  activeSeed?: Seed | null
  activeDomain?: Domain | null
  suggestions?: string[]
  error: string | null
  onDismissError: () => void
  onSend: (text: string) => void
  onPickContext: (ctx: ChatContext) => void
  queued?: QueuedMessage[]
  onCancelQueued?: (id: string) => void
  isChatStreaming?: boolean
  onStopChat?: () => void
  onAutoRunIdea?: (ideaId: string) => void
  autoResumable?: boolean   // the idea already has results → label the button "Resume"
  experiments?: ExperimentJob[]
  onJumpExperiment?: (job: ExperimentJob) => void
  autoRun?: AutoRunStatus | null
  onStopAutoRun?: () => void
}

export default function MainPanel({
  messages, activeIdea, activeSeed, activeDomain, suggestions,
  error, onDismissError, onSend, onPickContext,
  queued, onCancelQueued,
  isChatStreaming, onStopChat, onAutoRunIdea, autoResumable,
  experiments, onJumpExperiment, autoRun, onStopAutoRun,
}: Props) {
  // An auto run is in progress for THIS idea (banner shows its progress + stop).
  const autoRunningHere = autoRun?.status === 'running' && !!activeIdea && autoRun.ideaId === activeIdea.id
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const last = messages[messages.length - 1]
  const pendingQuestion =
    last && last.role === 'assistant' && last.status !== 'streaming' && last.question
      ? last.question
      : null

  return (
    <main className={s.panel}>
      <div className={s.topBar}>
        {activeDomain ? (
          <>
            <span className={s.topBarDot} />
            <span className={s.topBarTopic}>🌐 Domain: {activeDomain.name}</span>
          </>
        ) : activeSeed ? (
          <>
            <span className={s.topBarDot} />
            <span className={s.topBarTopic}>📝 Draft: {activeSeed.text}</span>
          </>
        ) : activeIdea ? (
          <>
            <span className={s.topBarDot} />
            <span className={s.topBarTopic}>{activeIdea.title}</span>
          </>
        ) : (
          <span className={s.topBarTopic} style={{ color: 'var(--text-disabled)' }}>
            Scratch conversation
          </span>
        )}
        <div className={s.topBarActions}>
          {activeIdea && (
            autoRunningHere ? (
              <span className={s.autoPhaseTag}>⚡ Auto running…</span>
            ) : (
              <button
                className={s.autoBtn}
                onClick={() => onAutoRunIdea?.(activeIdea.id)}
                title={autoResumable
                  ? 'Resume the auto pipeline on this idea — continues from existing results (untested hypotheses, then the paper).'
                  : 'Auto run this idea: hypotheses → experiments → paper (configure & run detached; tracked in the banner above). Ideas run in parallel.'}
              >
                {autoResumable ? '▶ Resume' : '⚡ Auto run'}
              </button>
            )
          )}
          <ExperimentMonitor jobs={experiments ?? []} onJump={j => onJumpExperiment?.(j)} />
          <HistoryMenu onPick={onPickContext} isStreaming={isChatStreaming} />
        </div>
      </div>

      {error && (
        <div className={s.errorBanner}>
          <span>⚠️ {error}</span>
          <button onClick={onDismissError} title="Dismiss">×</button>
        </div>
      )}

      {autoRun && activeIdea && autoRun.ideaId === activeIdea.id && (
        <AutoRunBanner autoRun={autoRun} onStop={onStopAutoRun} />
      )}

      <div className={s.messages}>
        {messages.length === 0 ? (
          <div className={s.welcome}>
            {activeDomain
              ? <div className={s.welcomeIcon}>🌐</div>
              : <img className={s.welcomeMascot} src="/mascot.png" alt="PaperClaw" />}
            <h1 className={s.welcomeTitle}>{activeDomain ? activeDomain.name : 'PaperClaw'}</h1>
            <p className={s.welcomeSub}>
              {activeDomain
                ? 'Discuss this domain or revise its spec — tailored starters below.'
                : 'Idea-oriented research assistant. Create a Domain first (left sidebar) — brainstorming digests its spec to draft ideas you can refine and pin.'}
            </p>
            <div className={s.suggestions}>
              {(suggestions ?? SUGGESTIONS).map(s2 => (
                <button key={s2} className={s.suggestionChip} onClick={() => onSend(s2)}>
                  {s2}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map(m => <MessageBubble key={m.id} message={m} />)}
            <div ref={bottomRef} style={{ height: 1 }} />
          </>
        )}
      </div>

      {pendingQuestion && <QuestionPanel question={pendingQuestion} onAnswer={onSend} />}

      {queued && queued.length > 0 && (
        <div className={s.queuedStrip}>
          <span className={s.queuedLabel}>⏳ Queued · sends when the reply finishes</span>
          {queued.map(q => (
            <div key={q.id} className={s.queuedItem}>
              <span className={s.queuedText}>{q.text}</span>
              <button
                className={s.queuedCancel}
                onClick={() => onCancelQueued?.(q.id)}
                title="Remove from queue"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {isChatStreaming && onStopChat && (
        <div className={s.stopStrip}>
          <button className={s.stopBtn} onClick={onStopChat} title="Stop generating this reply">
            ⏹ Stop generating
          </button>
        </div>
      )}

      <InputArea onSend={onSend} ideaActive={!!activeIdea} />
    </main>
  )
}
