import type { AutoRunStatus } from '../../types'
import s from './styles.module.css'

interface Props {
  autoRun?: AutoRunStatus | null
  onStop?: () => void
}

// The pipeline phases in order, with their display labels.
const PHASES: { key: string; label: string }[] = [
  { key: 'doctor', label: 'Doctor' },
  { key: 'domain', label: 'Domain' },
  { key: 'idea', label: 'Idea' },
  { key: 'hypotheses', label: 'Hypotheses' },
  { key: 'paper', label: 'Paper' },
]

/** A compact progress strip for an `paperclaw auto` run, so a CLI-launched run's
 *  overall phase (doctor → domain → idea → hypotheses → paper) is visible in the UI. */
export default function AutoRunBanner({ autoRun, onStop }: Props) {
  if (!autoRun) return null
  const { status, phase, label } = autoRun
  // Index of the active phase ('done' lands past the end so all show complete).
  const activeIdx = phase === 'done' ? PHASES.length : PHASES.findIndex(p => p.key === phase)

  const dot = status === 'running' ? '⚡' : status === 'error' ? '⚠️'
    : status === 'stopped' ? '⏸' : status === 'interrupted' ? '⚠' : '✓'
  return (
    <div className={`${s.autoRunBanner} ${status === 'error' ? s.autoRunError : ''}`}>
      <div className={s.autoRunHead}>
        <span className={s.autoRunTitle}>
          {dot} Auto run{autoRun.topic ? <> · <em>{autoRun.topic}</em></> : null}
        </span>
        {phase === 'hypotheses' || phase === 'paper' || phase === 'done' ? (
          <span className={s.autoRunCounts}>
            round {autoRun.round}/{autoRun.maxHypotheses} · {autoRun.positives}/{autoRun.targetPositive} positive
            {autoRun.currentHypothesisId ? ` · ${autoRun.currentHypothesisId}` : ''}
          </span>
        ) : null}
        {status === 'running' && onStop && (
          <button className={s.autoRunStop} onClick={onStop} title="Stop the auto pipeline">
            ⏹ Stop
          </button>
        )}
      </div>
      <div className={s.autoRunSteps}>
        {PHASES.map((p, i) => {
          const state = i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'todo'
          return (
            <span key={p.key} className={`${s.autoRunStep} ${s[`autoRunStep_${state}`]}`}>
              {state === 'done' ? '✓ ' : state === 'active' && status === 'running' ? '⚡ ' : ''}
              {p.label}
            </span>
          )
        })}
      </div>
      <div className={s.autoRunLabel}>
        {status === 'error' ? `⚠️ ${autoRun.error || label}`
          : status === 'stopped' ? '⏸ Stopped — resume with: paperclaw resume'
          : status === 'interrupted' ? '⚠ Interrupted (process gone) — resume with: paperclaw resume'
          : autoRun.paperReady ? '📄 Paper ready' : label}
      </div>
    </div>
  )
}
