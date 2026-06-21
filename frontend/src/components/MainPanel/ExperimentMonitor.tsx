import { useState } from 'react'
import type { ExperimentJob } from '../../types'
import s from './styles.module.css'

const ICON: Record<string, string> = {
  running: '▶', done: '✓', error: '✗', cancelled: '■', interrupted: '⚠',
}

function ago(ts: number): string {
  const sec = Math.max(0, Math.floor(Date.now() / 1000 - ts))
  if (sec < 60) return `${sec}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m`
  return `${Math.floor(sec / 3600)}h`
}

/** Group jobs by idea, preserving the (newest-first) order they arrive in. */
function groupByIdea(jobs: ExperimentJob[]) {
  const groups: { ideaId: string; ideaTitle: string; jobs: ExperimentJob[] }[] = []
  const index = new Map<string, number>()
  for (const j of jobs) {
    let gi = index.get(j.ideaId)
    if (gi === undefined) {
      gi = groups.length
      index.set(j.ideaId, gi)
      groups.push({ ideaId: j.ideaId, ideaTitle: j.ideaTitle, jobs: [] })
    }
    groups[gi].jobs.push(j)
  }
  return groups
}

/** Top-bar monitor of experiment jobs, structured as idea → its experiments.
 *  Click an idea to open it, or a hypothesis row to jump to its experiment. */
export default function ExperimentMonitor({ jobs, onJump }: {
  jobs: ExperimentJob[]
  onJump: (job: ExperimentJob) => void
}) {
  const [open, setOpen] = useState(false)
  const running = jobs.filter(j => j.status === 'running').length
  const groups = groupByIdea(jobs)

  return (
    <div className={s.expMonitor}>
      <button
        className={`${s.expPill} ${running ? s.expPillActive : ''}`}
        onClick={() => setOpen(o => !o)}
        title="Experiment jobs — running & recent, grouped by idea"
      >
        🧪 {running ? `${running} running` : jobs.length ? `${jobs.length}` : 'Experiments'}
      </button>
      {open && (
        <div className={s.expDropdown} onMouseLeave={() => setOpen(false)}>
          {!jobs.length && <div className={s.expEmpty}>No experiments yet — run one from a hypothesis's Experiment tab.</div>}
          {groups.map(g => {
            const grun = g.jobs.filter(j => j.status === 'running').length
            return (
              <div key={g.ideaId} className={s.expGroup}>
                <button
                  className={s.expIdeaHead}
                  onClick={() => { setOpen(false); onJump(g.jobs[0]) }}
                  title="Open this idea"
                >
                  <span className={s.expIdeaTitle}>{g.ideaTitle || g.ideaId}</span>
                  <span className={s.expIdeaMeta}>
                    {grun ? <span className={s.expRunDot}>▶ {grun}</span> : null} {g.jobs.length}
                  </span>
                </button>
                {g.jobs.map(j => (
                  <button
                    key={`${j.ideaId}/${j.hypothesisId}`}
                    className={s.expRow}
                    onClick={() => { setOpen(false); onJump(j) }}
                    title="Open this hypothesis's experiment"
                  >
                    <span className={`${s.expStatus} ${s['exp_' + j.status]}`}>{ICON[j.status] ?? '·'}</span>
                    <span className={s.expHid}>{j.hypothesisId}</span>
                    <span className={s.expAge}>{ago(j.startedAt)}</span>
                  </button>
                ))}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
