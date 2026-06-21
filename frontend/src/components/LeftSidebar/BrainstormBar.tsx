import { useState, useRef, KeyboardEvent } from 'react'
import type { BrainstormEmphasis, BrainstormIdeaType, BrainstormOptions, Seed } from '../../types'
import s from './styles.module.css'

interface Props {
  seeds: Seed[]
  streamingContexts: Set<string>
  isOpen: boolean
  generating: boolean
  generateStatus: string | null
  activeSeedId: string | null
  onToggle: () => void
  onAdd: (text: string) => void
  onRemove: (id: string) => void
  onGenerate: () => void
  opts: BrainstormOptions
  onOptsChange: (opts: BrainstormOptions) => void
  onOpen: (seed: Seed) => void
  onPromote: (seed: Seed) => void
}

const IDEA_TYPE_OPTS: { key: BrainstormIdeaType; label: string }[] = [
  { key: 'application', label: 'Application' },
  { key: 'algorithm', label: 'Algorithm' },
  { key: 'analysis', label: 'Analysis' },
  { key: 'benchmark', label: 'Benchmark' },
]

const EMPHASIS_OPTS: { key: BrainstormEmphasis; label: string }[] = [
  { key: 'performance', label: 'Performance' },
  { key: 'efficiency', label: 'Efficiency' },
  { key: 'robustness', label: 'Robustness' },
  { key: 'interpretability', label: 'Interpretability' },
]

const MAX_COUNT = 12

function toggleItem<T>(arr: T[], v: T): T[] {
  return arr.includes(v) ? arr.filter(x => x !== v) : [...arr, v]
}

export default function BrainstormBar({
  seeds, streamingContexts, isOpen, generating, generateStatus, activeSeedId,
  onToggle, onAdd, onRemove, onGenerate, opts, onOptsChange, onOpen, onPromote
}: Props) {
  const [draft, setDraft] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const setCount = (n: number) =>
    onOptsChange({ ...opts, count: Math.max(1, Math.min(MAX_COUNT, n)) })

  const commit = () => {
    if (!draft.trim()) return
    onAdd(draft)
    setDraft('')
  }

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.preventDefault(); commit() }
    if (e.key === 'Escape') { setDraft(''); inputRef.current?.blur() }
  }

  return (
    <>
      <div
        className={`${s.sectionHeader} ${isOpen ? s.active : ''}`}
        onClick={onToggle}
        role="button"
        aria-expanded={isOpen}
      >
        <span className={s.sectionIcon}>⚡</span>
        <span className={s.sectionTitle}>Brainstorm</span>
        {seeds.length > 0 && (
          <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginRight: 4 }}>
            {seeds.length}
          </span>
        )}
        <div className={s.sectionActions} onClick={e => e.stopPropagation()}>
          <button
            className={s.iconBtn}
            title="Brainstorm settings"
            onClick={() => { setSettingsOpen(v => !v); if (!isOpen) onToggle() }}
          >
            ⚙
          </button>
          <button
            className={s.iconBtn}
            title="Brainstorm from selected domains"
            disabled={generating}
            onClick={() => { onGenerate(); if (!isOpen) onToggle() }}
          >
            {generating ? '…' : '✨'}
          </button>
        </div>
        <span className={`${s.chevron} ${isOpen ? s.chevronOpen : ''}`}>›</span>
      </div>

      {isOpen && (
        <div className={s.sectionBody}>
          {settingsOpen && (
            <div className={s.bsSettings}>
              <div className={s.bsGroup}>
                <span className={s.bsLabel}>Idea type</span>
                <div className={s.bsChips}>
                  {IDEA_TYPE_OPTS.map(o => (
                    <button
                      key={o.key}
                      className={`${s.bsChip} ${opts.ideaTypes.includes(o.key) ? s.bsChipOn : ''}`}
                      onClick={() => onOptsChange({ ...opts, ideaTypes: toggleItem(opts.ideaTypes, o.key) })}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className={s.bsGroup}>
                <span className={s.bsLabel}>Emphasis</span>
                <div className={s.bsChips}>
                  {EMPHASIS_OPTS.map(o => (
                    <button
                      key={o.key}
                      className={`${s.bsChip} ${opts.emphasis.includes(o.key) ? s.bsChipOn : ''}`}
                      onClick={() => onOptsChange({ ...opts, emphasis: toggleItem(opts.emphasis, o.key) })}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className={s.bsGroup}>
                <span className={s.bsLabel}>Count</span>
                <div className={s.bsStepper}>
                  <button className={s.bsStepBtn} onClick={() => setCount(opts.count - 1)} disabled={opts.count <= 1}>−</button>
                  <span className={s.bsCount}>{opts.count}</span>
                  <button className={s.bsStepBtn} onClick={() => setCount(opts.count + 1)} disabled={opts.count >= MAX_COUNT}>+</button>
                </div>
              </div>
              <p className={s.bsHint}>Unchecked = anything goes. Applies to ✨ generation.</p>
            </div>
          )}
          <div className={s.ideaInput}>
            <input
              ref={inputRef}
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={onKey}
              placeholder="Capture a seed…"
              maxLength={280}
            />
            <button className={s.addBtn} onClick={commit} title="Add (Enter)">+</button>
          </div>

          {seeds.length === 0 ? (
            <p className={s.emptyHint}>
              {generating
                ? (generateStatus ?? '✨ Brainstorming ideas…')
                : 'No seeds yet — select domains and press ✨'}
            </p>
          ) : (
            <div className={s.ideaList}>
              {generating && (
                <p className={s.emptyHint}>{generateStatus ?? '✨ Brainstorming ideas…'}</p>
              )}
              {seeds.map(seed => (
                <div
                  key={seed.id}
                  className={`${s.ideaCard} ${seed.id === activeSeedId ? s.ideaCardActive : ''} ${seed.draft ? s.ideaCardClickable : ''}`}
                  onClick={() => seed.draft && onOpen(seed)}
                  title={seed.draft ? 'Click to discuss this draft' : undefined}
                >
                  <span className={s.ideaDot} />
                  <span className={s.ideaText}>
                    {seed.draft && <span className={s.draftBadge}>📝</span>}
                    {seed.text}
                  </span>
                  {streamingContexts.has(`seed-${seed.id}`) && (
                    <span className={s.itemSpinner} title="Generating…" />
                  )}
                  <button
                    className={s.ideaPromote}
                    onClick={e => { e.stopPropagation(); onPromote(seed) }}
                    title="Pin as Idea (creates IDEA.md)"
                  >
                    ↗
                  </button>
                  <button
                    className={s.ideaDelete}
                    onClick={e => { e.stopPropagation(); onRemove(seed.id) }}
                    title="Remove"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  )
}
