import { useState } from 'react'
import s from './styles.module.css'

interface Settings { topic: string; positive: number; maxHypotheses: number; pageLimit: number; maxDepth: number }

interface Props {
  open: boolean
  onClose: () => void
  onStart: (settings: Settings) => void
  running?: boolean  // a run is already in progress
}

function Field({ label, value, min, max, onChange }:
  { label: string; value: number; min: number; max: number; onChange: (n: number) => void }) {
  return (
    <label className={s.field}>
      <span>{label}</span>
      <input type="number" min={min} max={max} value={value}
             onChange={e => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))} />
    </label>
  )
}

/** Modal to configure + launch an `auto` run (topic + key settings) from the web UI. */
export default function AutoRunLauncher({ open, onClose, onStart, running }: Props) {
  const [topic, setTopic] = useState('')
  const [positive, setPositive] = useState(2)
  const [maxHypotheses, setMaxHypotheses] = useState(6)
  const [pageLimit, setPageLimit] = useState(8)
  const [maxDepth, setMaxDepth] = useState(3)
  if (!open) return null

  const start = () => {
    const t = topic.trim()
    if (t && !running) onStart({ topic: t, positive, maxHypotheses, pageLimit, maxDepth })
  }

  return (
    <div className={s.overlay} onClick={onClose}>
      <div className={s.modal} onClick={e => e.stopPropagation()}>
        <h2 className={s.title}>⚡ Auto run</h2>
        <p className={s.sub}>
          Runs the whole pipeline from a topic: doctor → domain → idea → hypotheses → paper.
        </p>
        {running && <div className={s.warn}>A run is already in progress — stop it first.</div>}

        <label className={s.label}>Topic</label>
        <textarea
          className={s.topic} value={topic} onChange={e => setTopic(e.target.value)}
          placeholder="e.g. generative modeling for time series" rows={2} autoFocus
          onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) start() }}
        />

        <div className={s.row}>
          <Field label="Paper on N positive" value={positive} min={1} max={10} onChange={setPositive} />
          <Field label="Max hypotheses" value={maxHypotheses} min={1} max={20} onChange={setMaxHypotheses} />
          <Field label="Max tree depth" value={maxDepth} min={1} max={6} onChange={setMaxDepth} />
          <Field label="Page limit" value={pageLimit} min={1} max={20} onChange={setPageLimit} />
        </div>

        <div className={s.actions}>
          <button className={s.cancel} onClick={onClose}>Cancel</button>
          <button className={s.start} disabled={!topic.trim() || running} onClick={start}>
            Start run
          </button>
        </div>
      </div>
    </div>
  )
}
