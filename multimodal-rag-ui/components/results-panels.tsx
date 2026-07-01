'use client'

import { ImageItem, TableItem } from '@/lib/ragService'

interface ResultsPanelsProps {
  images: ImageItem[]
  tables: TableItem[]
}

// Convert Windows path from DB to a URL served by FastAPI /images endpoint
function toImageUrl(imagePath: string | null): string | null {
  if (!imagePath) return null
  // imagePath example:
  // data\images\BERT-Pre-training...\BERT-Pre-training..._p3_img0.png
  // We need everything AFTER "data\images\" to build the URL
  const normalized = imagePath.replace(/\\/g, '/')
  const marker = 'data/images/'
  const idx = normalized.indexOf(marker)
  if (idx !== -1) {
    const relative = normalized.slice(idx + marker.length)
    return `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/images/${relative}`
  }
  // fallback — just use filename
  const parts = normalized.split('/')
  return `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/images/${parts[parts.length - 1]}`
}

export default function ResultsPanels({ images, tables }: ResultsPanelsProps) {
  return (
    <div className="grid grid-cols-2 gap-6 w-full">

      {/* Image Panel */}
      {images.length > 0 && (
        <div className="bg-card border border-border rounded-2xl overflow-hidden">
          {images.map((img, i) => {
            const url = toImageUrl(img.image_path)
            return (
              <div key={i}>
                <div className="bg-gradient-to-b from-secondary/20 to-secondary/5
                                aspect-video flex items-center justify-center p-4">
                  {url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={url}
                      alt={img.description}
                      className="max-h-full max-w-full object-contain rounded"
                      onError={(e) => {
                        // Hide broken image, show description instead
                        (e.target as HTMLImageElement).style.display = 'none'
                      }}
                    />
                  ) : (
                    <div className="text-center space-y-2">
                      <svg className="w-8 h-8 text-muted-foreground mx-auto"
                           fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      <p className="text-xs text-muted-foreground">Image not available</p>
                    </div>
                  )}
                </div>
                <div className="p-4 border-t border-border">
                  <h3 className="font-semibold text-foreground text-sm mb-1">Retrieved Image</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">
                    {img.description}
                  </p>
                  <div className="mt-3 flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">{img.doc_id} · Page {img.page}</span>
                    <span className="text-primary font-semibold">
                      {Math.round(img.similarity * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Table Panel */}
      {tables.length > 0 && (
        <div className="bg-card border border-border rounded-2xl overflow-hidden flex flex-col">
          {tables.map((tbl, i) => {
            let rows: string[][] = []
            try {
              if (tbl.raw_table) rows = JSON.parse(tbl.raw_table)
            } catch { /* use description fallback */ }

            return (
              <div key={i}>
                <div className="p-4 border-b border-border">
                  <h3 className="font-semibold text-foreground text-sm mb-1">Retrieved Table</h3>
                  <p className="text-xs text-muted-foreground line-clamp-2">{tbl.description}</p>
                </div>

                {rows.length > 0 ? (
                  <div className="overflow-x-auto flex-1">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border bg-secondary/20">
                          {rows[0].map((header, hi) => (
                            <th key={hi} className="px-4 py-2 text-left font-semibold
                                                    text-foreground whitespace-nowrap text-xs">
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.slice(1).map((row, ri) => (
                          <tr key={ri} className="border-b border-border/50
                                                  hover:bg-secondary/10 transition-colors">
                            {row.map((cell, ci) => (
                              <td key={ci} className="px-4 py-2 text-foreground text-xs">
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="p-4 text-sm text-muted-foreground">{tbl.description}</div>
                )}

                <div className="p-4 border-t border-border text-xs text-muted-foreground">
                  {tbl.doc_id} · Page {tbl.page} · {Math.round(tbl.similarity * 100)}% relevance
                </div>
              </div>
            )
          })}
        </div>
      )}

    </div>
  )
}