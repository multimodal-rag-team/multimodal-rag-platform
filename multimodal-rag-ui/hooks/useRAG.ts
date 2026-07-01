"use client";

import { useState, useEffect, useCallback } from "react";
import {
  queryDocuments,
  listDocuments,
  indexFiles,
  checkHealth,
  QueryResponse,
  DocumentInfo,
  IndexResult,
} from "@/lib/ragService";

export function useQuery() {
  const [result, setResult]   = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const submit = useCallback(
    async (query: string, docId: string | null = null, topK = 5) => {
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        const data = await queryDocuments(query, docId, topK);
        setResult(data);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Query failed.");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { submit, result, loading, error };
}

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch {
      setError("Failed to load documents.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  return { documents, loading, error, refetch: fetchDocs };
}

export function useIndexFiles(onSuccess?: (results: IndexResult[]) => void) {
  const [results, setResults] = useState<IndexResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      setLoading(true);
      setError(null);
      try {
        const indexed = await indexFiles(Array.from(files));
        setResults(indexed);
        // ── KEY FIX: call onSuccess AFTER indexing completes ──
        if (onSuccess) onSuccess(indexed);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Upload failed.");
      } finally {
        // ── KEY FIX: always stop spinner ──
        setLoading(false);
      }
    },
    [onSuccess]
  );

  return { upload, results, loading, error };
}

export function useHealth() {
  const [isOnline, setIsOnline] = useState(false);

  useEffect(() => {
    const check = async () => setIsOnline(await checkHealth());
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, []);

  return { isOnline };
}