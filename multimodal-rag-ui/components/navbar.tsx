import { Brain, Circle } from 'lucide-react'

interface NavbarProps {
  isOnline: boolean
}

export default function Navbar({ isOnline }: NavbarProps) {
  return (
    <nav className="border-b border-border bg-card/50 backdrop-blur-lg">
      <div className="h-16 px-8 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-primary to-primary/70 rounded-lg">
            <Brain className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-foreground">MultiModal RAG</h1>
            <p className="text-xs text-muted-foreground">Retrieval-Augmented Generation</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Circle className={`w-2.5 h-2.5 ${isOnline ? 'fill-green-500 text-green-500' : 'fill-red-500 text-red-500'}`} />
          <span className="text-sm font-medium text-muted-foreground">
            {isOnline ? 'Ready' : 'Offline'}
          </span>
        </div>
      </div>
    </nav>
  )
}
