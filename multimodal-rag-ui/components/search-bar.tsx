'use client'

import { useState } from 'react'
import { Search, ArrowRight } from 'lucide-react'

interface SearchBarProps {
  onSearch: (query: string) => void
  initialValue?: string
}

export default function SearchBar({ onSearch, initialValue = '' }: SearchBarProps) {
  const [query, setQuery] = useState(initialValue)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-accent/20 rounded-2xl blur opacity-0 group-hover:opacity-100 transition duration-300"></div>
        <div className="relative flex items-center gap-4 bg-card border border-border/50 rounded-2xl px-6 py-4 hover:border-border transition-colors group">
          <Search className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors flex-shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask anything about your documents..."
            className="flex-1 bg-transparent text-foreground placeholder-muted-foreground focus:outline-none text-base"
          />
          <button
            type="submit"
            disabled={!query.trim()}
            className="p-2 bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground text-primary-foreground rounded-lg transition-colors flex-shrink-0"
          >
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </div>
    </form>
  )
}
