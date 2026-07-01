'use client'

import { SourceItem } from '@/lib/ragService'

interface AnswerCardProps {
  answer: string
  sources: SourceItem[]
}

export default function AnswerCard({ answer, sources }: AnswerCardProps) {
  return (
    <div className="w-full bg-card border border-border rounded-2xl p-8 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-bold text-foreground mb-2">Answer</h2>
          <p className="text-sm text-muted-foreground">Based on retrieved documents</p>
        </div>
      </div>

      {/* Answer Text */}
      <div className="text-base leading-relaxed text-foreground/90 whitespace-pre-wrap">
        {answer}
      </div>

      {/* Citations */}
      {sources.length > 0 && (
        <div className="pt-6 border-t border-border">
          <p className="text-xs text-muted-foreground mb-3 uppercase tracking-wide">Citations</p>
          <div className="flex flex-wrap gap-2">
            {sources.map((src, i) => (
              <div
                key={i}
                className="inline-flex items-center gap-2 bg-secondary/40 hover:bg-secondary/60
                           px-3 py-2 rounded-lg transition-colors cursor-pointer group"
              >
                <span className="text-sm font-semibold text-primary">[{i + 1}]</span>
                <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                  {src.doc_id} · p{src.page}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
