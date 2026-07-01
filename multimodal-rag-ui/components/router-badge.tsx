'use client'

import { FileText, Image, Table, Zap } from 'lucide-react'

interface RouterBadgeProps {
  route: 'text' | 'table' | 'image' | 'hybrid'
  reasoning?: string
}

const ROUTE_CONFIG = {
  text:   { label: 'Text',   icon: FileText, bgColor: 'bg-blue-500/10',   textColor: 'text-blue-400' },
  table:  { label: 'Table',  icon: Table,    bgColor: 'bg-amber-500/10',  textColor: 'text-amber-400' },
  image:  { label: 'Image',  icon: Image,    bgColor: 'bg-green-500/10',  textColor: 'text-green-400' },
  hybrid: { label: 'Hybrid', icon: Zap,      bgColor: 'bg-purple-500/10', textColor: 'text-purple-400' },
}

export default function RouterBadge({ route, reasoning }: RouterBadgeProps) {
  const config = ROUTE_CONFIG[route] || ROUTE_CONFIG.text
  const Icon = config.icon

  return (
    <div className="flex items-center gap-4">
      <span className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        Query Router:
      </span>
      <div className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-semibold text-sm
        ${config.bgColor} ${config.textColor} border border-current shadow-lg`}>
        <Icon className="w-4 h-4" />
        <span>{config.label}</span>
        <span className="w-2 h-2 rounded-full bg-current" />
      </div>
      {reasoning && (
        <p className="text-xs text-muted-foreground italic truncate max-w-xs">{reasoning}</p>
      )}
    </div>
  )
}
