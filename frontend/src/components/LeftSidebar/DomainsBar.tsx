import { useState, KeyboardEvent } from 'react'
import type { Domain } from '../../types'
import s from './styles.module.css'

interface Props {
  domains: Domain[]
  streamingContexts: Set<string>
  isOpen: boolean
  autoCreating: boolean
  autoCreateStatus: string | null
  viewDomainId: string | null
  onToggle: () => void
  onAuto: (prompt: string) => void
  onStartWizard: () => void
  onSelect: (id: string, selected: boolean) => void
  onView: (id: string) => void
  onRemove: (id: string) => void
  onSetCodebase: (id: string, url: string) => void
}

export default function DomainsBar({
  domains, streamingContexts, isOpen, autoCreating, autoCreateStatus, viewDomainId,
  onToggle, onAuto, onStartWizard, onSelect, onView, onRemove, onSetCodebase
}: Props) {
  const [adding, setAdding] = useState(false)
  const [autoPrompt, setAutoPrompt] = useState<string | null>(null)

  const commitAuto = () => {
    if (autoPrompt?.trim()) onAuto(autoPrompt.trim())
    setAutoPrompt(null)
    setAdding(false)
  }

  const onAutoKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.preventDefault(); commitAuto() }
    if (e.key === 'Escape') { setAutoPrompt(null); setAdding(false) }
  }

  return (
    <>
      <div
        className={`${s.sectionHeader} ${isOpen ? s.active : ''}`}
        onClick={onToggle}
        role="button"
        aria-expanded={isOpen}
      >
        <span className={s.sectionIcon}>🌐</span>
        <span className={s.sectionTitle}>Domains</span>
        {domains.length > 0 && (
          <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginRight: 4 }}>
            {domains.filter(d => d.isSelected).length}/{domains.length}
          </span>
        )}
        <div className={s.sectionActions} onClick={e => e.stopPropagation()}>
          <button
            className={s.iconBtn}
            title="New domain"
            onClick={() => { setAdding(v => !v); if (!isOpen) onToggle() }}
          >
            +
          </button>
        </div>
        <span className={`${s.chevron} ${isOpen ? s.chevronOpen : ''}`}>›</span>
      </div>

      {isOpen && (
        <div className={s.sectionBody}>
          {adding && autoPrompt === null && (
            <div className={s.domainModeRow}>
              <button
                className={s.domainModeBtn}
                disabled={autoCreating}
                onClick={() => setAutoPrompt('')}
                title="The LLM writes the whole DOMAIN.md from a short prompt"
              >
                ✨ Auto
              </button>
              <button
                className={s.domainModeBtn}
                onClick={() => { setAdding(false); onStartWizard() }}
                title="Step-by-step questions in chat (/create_domain)"
              >
                🧭 Guided
              </button>
            </div>
          )}

          {autoPrompt !== null && (
            <div className={s.ideaInput}>
              <input
                autoFocus
                value={autoPrompt}
                onChange={e => setAutoPrompt(e.target.value)}
                onKeyDown={onAutoKey}
                placeholder="Describe the domain… (Enter)"
                maxLength={300}
              />
              <button className={s.addBtn} onClick={commitAuto} title="Create (Enter)">✨</button>
            </div>
          )}

          {autoCreating && (
            <p className={s.emptyHint}>
              {autoCreateStatus ?? '✨ Building domain spec…'}
            </p>
          )}

          {domains.length === 0 && !adding && !autoCreating ? (
            <p className={s.emptyHint}>No domains yet — press + to create one</p>
          ) : (
            <div className={s.topicList}>
              {domains.map(domain => (
                <div
                  key={domain.id}
                  className={`${s.topicItem} ${viewDomainId === domain.id ? s.topicActive : ''}`}
                  onClick={() => onView(domain.id)}
                  role="button"
                  title="Click to view DOMAIN.md"
                >
                  <input
                    type="checkbox"
                    className={s.domainCheck}
                    checked={domain.isSelected}
                    onChange={e => onSelect(domain.id, e.target.checked)}
                    onClick={e => e.stopPropagation()}
                    title="Use this domain for brainstorming"
                  />
                  <span className={s.topicName} title={domain.name}>{domain.name}</span>
                  {streamingContexts.has(`domain-${domain.id}`) && (
                    <span className={s.itemSpinner} title="Generating…" />
                  )}
                  <button
                    className={s.topicDelete}
                    onClick={e => {
                      e.stopPropagation()
                      const cur = domain.codebaseUrl ?? ''
                      const url = window.prompt(
                        'Reference codebase — GitHub repo URL for this domain\n'
                        + '(experiments here reuse it). Blank to clear.',
                        cur)
                      if (url === null) return
                      onSetCodebase(domain.id, url.trim())
                    }}
                    title={domain.codebaseFiles
                      ? `Reference codebase: ${domain.codebaseUrl} (${domain.codebaseFiles} files) — click to change`
                      : 'Set a reference codebase (GitHub repo) for experiments'}
                  >
                    {domain.codebaseFiles ? `📦${domain.codebaseFiles}` : '📦'}
                  </button>
                  <button
                    className={s.topicDelete}
                    onClick={e => { e.stopPropagation(); onRemove(domain.id) }}
                    title="Remove (deletes DOMAIN.md)"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      <div className={s.divider} />
    </>
  )
}
