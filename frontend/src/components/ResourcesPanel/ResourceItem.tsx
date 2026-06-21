import type { Resource } from '../../types'
import s from './styles.module.css'

interface Props { resource: Resource }

const TYPE_LABEL: Record<string, string> = {
  paper: 'Paper', link: 'Link', dataset: 'Dataset', code: 'Code'
}

export default function ResourceItem({ resource }: Props) {
  const rel = resource.relevance ?? 0
  const dots = [1, 2, 3, 4, 5]

  return (
    <div className={s.card}>
      <div className={s.cardTop}>
        <span className={`${s.typeTag} ${s[resource.type]}`}>
          {TYPE_LABEL[resource.type]}
        </span>
        <span className={s.cardTitle}>{resource.title}</span>
      </div>

      <div className={s.cardMeta}>
        {resource.authors && resource.authors.length > 0 && (
          <span>{resource.authors.slice(0, 2).join(', ')}{resource.authors.length > 2 ? ' et al.' : ''}</span>
        )}
        {resource.year && (
          <>
            <span className={s.metaDot} />
            <span>{resource.year}</span>
          </>
        )}
        {resource.venue && (
          <>
            <span className={s.metaDot} />
            <span>{resource.venue}</span>
          </>
        )}
        {rel > 0 && (
          <div className={s.relevance} title={`Relevance: ${rel}/5`}>
            {dots.map(d => (
              <span key={d} className={`${s.relDot} ${d <= rel ? s.filled : ''}`} />
            ))}
          </div>
        )}
      </div>

      {resource.summary && (
        <p style={{
          marginTop: 6,
          fontSize: 'var(--fs-xs)',
          color: 'var(--text-muted)',
          lineHeight: 1.5,
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden'
        }}>
          {resource.summary}
        </p>
      )}
    </div>
  )
}
