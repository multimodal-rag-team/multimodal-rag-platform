'use client'

import { FileText } from 'lucide-react'
import { SourceItem } from '@/lib/ragService'

interface SourcePanelProps {
  sources: SourceItem[]
}

export default function SourcePanel({ sources }: SourcePanelProps) {
  return (
    <div className="w-full bg-card border border-border rounded-2xl p-8">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-foreground mb-2">Retrieved Sources</h2>
        <p className="text-sm text-muted-foreground">Documents ranked by relevance score</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sources.map((src, i) => {
          const similarityPct = Math.round(src.similarity * 100)
          return (
            <div
              key={i}
              className="bg-gradient-to-br from-secondary/10 to-secondary/5 border border-border/50
                         rounded-xl p-5 hover:border-primary/40 hover:bg-secondary/20 transition-all group"
            >
              <div className="flex items-start gap-3 mb-4">
                <div className="p-2.5 bg-primary/10 rounded-lg group-hover:bg-primary/20 transition-colors">
                  <FileText className="w-4 h-4 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-foreground text-sm group-hover:text-primary transition-colors truncate">
                    {src.doc_id}
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    Page {src.page} · {src.type}
                  </p>
                </div>
              </div>

              {/* Similarity Score */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground font-medium">Relevance</span>
                  <span className="text-primary font-bold">{similarityPct}%</span>
                </div>
                <div className="h-2 bg-secondary/50 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all duration-500"
                    style={{ width: `${similarityPct}%` }}
                  />
                </div>
              </div>

              {/* Content preview */}
              {src.content && (
                <p className="mt-3 text-xs text-muted-foreground line-clamp-2">{src.content}</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
