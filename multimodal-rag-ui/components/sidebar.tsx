'use client'

import { FileText, Image, Table, Grid3X3, Upload, Loader2 } from 'lucide-react'
import { DocumentInfo } from '@/lib/ragService'

interface SidebarProps {
  activeFilter: string
  onFilterChange: (filter: string) => void
  documents: DocumentInfo[]
  loading: boolean
  selectedDocId: string | null
  onDocumentSelect: (id: string) => void
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void
  uploading: boolean
}

export default function Sidebar({
  activeFilter,
  onFilterChange,
  documents,
  loading,
  selectedDocId,
  onDocumentSelect,
  onUpload,
  uploading,
}: SidebarProps) {
  const filters = [
    { id: 'all',   label: 'All Documents', icon: Grid3X3 },
    { id: 'text',  label: 'Text',          icon: FileText },
    { id: 'table', label: 'Tables',        icon: Table },
    { id: 'image', label: 'Images',        icon: Image },
  ]

  return (
    <aside className="w-64 border-r border-border bg-card/50 backdrop-blur-sm overflow-hidden flex flex-col">
      {/* Filter Section */}
      <div className="p-6 border-b border-border">
        <h2 className="text-sm font-semibold text-foreground mb-4">Filter by Type</h2>
        <div className="space-y-2">
          {filters.map((filter) => {
            const Icon = filter.icon
            const isActive = activeFilter === filter.id
            return (
              <button
                key={filter.id}
                onClick={() => onFilterChange(filter.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors font-medium text-sm ${
                  isActive
                    ? 'bg-primary/20 text-primary border border-primary/40'
                    : 'text-muted-foreground hover:bg-secondary/40 hover:text-foreground border border-transparent'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{filter.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Documents List */}
      <div className="flex-1 overflow-auto p-6">
        <h3 className="text-sm font-semibold text-foreground mb-4">Documents</h3>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : documents.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-8">
            No documents indexed yet. Upload PDFs below.
          </p>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.doc_id}
                onClick={() => onDocumentSelect(doc.doc_id)}
                className={`p-3 rounded-lg cursor-pointer transition-colors group ${
                  selectedDocId === doc.doc_id
                    ? 'bg-primary/20 border border-primary/40'
                    : 'bg-secondary/30 hover:bg-secondary/60 border border-transparent'
                }`}
              >
                <p className="text-sm font-medium text-foreground group-hover:text-primary transition-colors truncate">
                  {doc.doc_id}
                </p>
                <div className="flex gap-2 mt-1 text-xs text-muted-foreground">
                  <span>{doc.text_chunks}T</span>
                  <span>{doc.table_chunks}Tb</span>
                  <span>{doc.image_chunks}I</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Upload Button */}
      <div className="p-6 border-t border-border">
        <label className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg
          border border-dashed border-border cursor-pointer transition-colors text-sm font-medium
          ${uploading ? 'opacity-50 cursor-not-allowed' : 'hover:border-primary hover:text-primary text-muted-foreground'}`}>
          {uploading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Indexing...</>
            : <><Upload className="w-4 h-4" /> Upload PDFs</>
          }
          <input
            type="file"
            accept=".pdf"
            multiple
            className="hidden"
            onChange={onUpload}
            disabled={uploading}
          />
        </label>
      </div>
    </aside>
  )
}
