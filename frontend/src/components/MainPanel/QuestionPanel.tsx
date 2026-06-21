import { useState, KeyboardEvent } from 'react'
import type { ChatQuestion } from '../../types'
import s from './styles.module.css'

interface Props {
  question: ChatQuestion
  onAnswer: (text: string) => void
}

// Options that just mean "I'll type my own" are redundant with the free-text box,
// so we hide them when allowFreeText is on (and treat a click as a free-text submit).
const FREE_TEXT_OPTION = /let me (answer|type)|answer myself|type (my|your) own/i

export default function QuestionPanel({ question, onAnswer }: Props) {
  const [free, setFree] = useState('')
  const allowFree = question.allowFreeText !== false
  // Drop the redundant "answer myself" option when the free-text box is shown.
  const options = allowFree
    ? question.options.filter(o => !FREE_TEXT_OPTION.test(o))
    : question.options

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && free.trim()) {
      e.preventDefault()
      onAnswer(free.trim())
      setFree('')
    }
  }

  const handleOption = (opt: string) => {
    if (FREE_TEXT_OPTION.test(opt)) {
      if (free.trim()) { onAnswer(free.trim()); setFree('') }
    } else {
      onAnswer(opt)
    }
  }

  return (
    <div className={s.questionPanel}>
      <div className={s.questionPrompt}>
        <span className={s.questionIcon}>?</span>
        {question.prompt}
      </div>
      <div className={s.questionOptions}>
        {options.map((opt, i) => (
          <button
            key={i}
            className={s.questionOption}
            onClick={() => handleOption(opt)}
          >
            <span className={s.questionNum}>{i + 1}</span>
            {opt}
          </button>
        ))}
      </div>
      {allowFree && (
        <input
          className={s.questionFree}
          value={free}
          onChange={e => setFree(e.target.value)}
          onKeyDown={onKey}
          placeholder="Or type your own answer… (Enter)"
        />
      )}
    </div>
  )
}
