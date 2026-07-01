"use client";

import { useState } from "react";
import { useQuery, useDocuments, useIndexFiles, useHealth } from "@/hooks/useRAG";
import Navbar        from "@/components/navbar";
import Sidebar       from "@/components/sidebar";
import SearchBar     from "@/components/search-bar";
import AnswerCard    from "@/components/answer-card";
import ResultsPanels from "@/components/results-panels";
import RouterBadge   from "@/components/router-badge";
import SourcePanel   from "@/components/source-panel";

type FilterType = "all" | "text" | "table" | "image";

export default function Home() {
  const [activeFilter, setActiveFilter]   = useState<FilterType>("all");
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  const { isOnline }                         = useHealth();
  const { documents, loading: docsLoading,
          refetch: refetchDocs }             = useDocuments();
  const { submit, result, loading, error }   = useQuery();
  const { upload, loading: uploading }       = useIndexFiles(refetchDocs);

  // ── Derived results — filter by active type ───────────────────
  const allSources = result?.sources ?? [];
  const allImages  = result?.images  ?? [];
  const allTables  = result?.tables  ?? [];

  // Apply filter to results
  const sources = activeFilter === "image" || activeFilter === "table"
    ? [] : allSources;
  const images  = activeFilter === "text" || activeFilter === "table"
    ? [] : allImages;
  const tables  = activeFilter === "text" || activeFilter === "image"
    ? [] : allTables;

  // Show all when filter is "all"
  const visibleSources = activeFilter === "all" ? allSources : sources;
  const visibleImages  = activeFilter === "all" ? allImages  : images;
  const visibleTables  = activeFilter === "all" ? allTables  : tables;

  // ── Filter sidebar docs by active type ────────────────────────
  const filteredDocs = documents.filter((doc) => {
    if (activeFilter === "text")  return doc.text_chunks  > 0;
    if (activeFilter === "table") return doc.table_chunks > 0;
    if (activeFilter === "image") return doc.image_chunks > 0;
    return true;
  });

  const handleFilterChange = (filter: string) => {
    setActiveFilter(filter as FilterType);
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) upload(e.target.files);
  };

  return (
    <div className="flex flex-col h-screen bg-[#0d1117] text-white">
      <Navbar isOnline={isOnline} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          activeFilter={activeFilter}
          onFilterChange={handleFilterChange}
          documents={filteredDocs}
          loading={docsLoading}
          selectedDocId={selectedDocId}
          onDocumentSelect={(id: string) =>
            setSelectedDocId(id === selectedDocId ? null : id)
          }
          onUpload={handleUpload}
          uploading={uploading}
        />

        <main className="flex-1 flex flex-col items-center overflow-y-auto p-8 gap-6">

          <div className="w-full max-w-3xl">
            <SearchBar onSearch={(q: string) => submit(q, selectedDocId)} />
          </div>

          {/* Active filter indicator */}
          {activeFilter !== "all" && (
            <div className="w-full max-w-3xl flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Filtering results by:</span>
              <span className="text-xs font-semibold text-primary capitalize px-2 py-1
                               bg-primary/10 border border-primary/30 rounded-full">
                {activeFilter}
              </span>
              <button
                onClick={() => setActiveFilter("all")}
                className="text-xs text-muted-foreground hover:text-foreground ml-2"
              >
                Clear ×
              </button>
            </div>
          )}

          {loading && (
            <div className="w-full max-w-3xl space-y-3 animate-pulse">
              {[75, 55, 65].map((w, i) => (
                <div key={i} className="h-4 rounded bg-slate-700" style={{ width: `${w}%` }} />
              ))}
            </div>
          )}

          {error && (
            <div className="w-full max-w-3xl rounded-lg border border-red-500
                            bg-red-900/30 px-4 py-3 text-sm text-red-300">
              {error}
            </div>
          )}

          {result && !loading && (
            <div className="w-full max-w-3xl flex flex-col gap-5">
              <RouterBadge route={result.route} reasoning={result.reasoning} />
              <AnswerCard answer={result.answer} sources={visibleSources} />
              {(visibleImages.length > 0 || visibleTables.length > 0) && (
                <ResultsPanels images={visibleImages} tables={visibleTables} />
              )}
              {visibleSources.length > 0 && <SourcePanel sources={visibleSources} />}
            </div>
          )}

          {!result && !loading && !error && (
            <div className="flex flex-col items-center justify-center flex-1
                            gap-3 mt-20 text-slate-500">
              <p className="text-lg font-medium">Intelligent Multi-Modal Search</p>
              <p className="text-sm text-center max-w-sm">
                Search across documents, images, and tables with AI-powered understanding.
              </p>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}