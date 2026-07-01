const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface SourceItem {
  doc_id: string;
  page: number;
  type: string;
  similarity: number;
  content: string;
  metadata: Record<string, unknown>;
}

export interface TableItem {
  doc_id: string;
  page: number;
  similarity: number;
  description: string;
  raw_table: string | null;
  metadata: Record<string, unknown>;
}

export interface ImageItem {
  doc_id: string;
  page: number;
  similarity: number;
  description: string;
  image_path: string | null;
  metadata: Record<string, unknown>;
}

export interface QueryResponse {
  answer: string;
  route: "text" | "table" | "image" | "hybrid";
  reasoning: string;
  sources: SourceItem[];
  images: ImageItem[];
  tables: TableItem[];
}

export interface DocumentInfo {
  doc_id: string;
  text_chunks: number;
  table_chunks: number;
  image_chunks: number;
  last_indexed_at: string | null;
}

export interface IndexResult {
  doc_id: string;
  filename: string;
  counts: Record<string, number>;
}

export async function queryDocuments(
  query: string,
  doc_id: string | null = null,
  top_k: number = 5
): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, doc_id, top_k }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${BASE_URL}/documents`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.documents;
}

export async function indexFiles(files: File[]): Promise<IndexResult[]> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  // ── Increased timeout for large PDFs with many images ──
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min timeout

  try {
    const res = await fetch(`${BASE_URL}/index`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    return data.indexed;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    const data = await res.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}