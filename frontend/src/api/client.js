/**
 * FastAPI client for the RAG backend.
 *
 * Default: same-origin requests (Vite proxies /upload and /query in dev).
 * Override with VITE_API_URL (e.g. http://127.0.0.1:8000).
 */
const BASE_URL = import.meta.env.VITE_API_URL ?? "";

async function parseError(response) {
  let body;
  try {
    body = await response.json();
  } catch {
    body = await response.text();
  }
  const detail = body?.detail || body?.message || body;
  return new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
}

function withTimeout(ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, clear: () => clearTimeout(timer) };
}

export async function queryRag({ query, hybrid, hyde, top_k = 10 }) {
  const t = withTimeout(5 * 60 * 1000);
  try {
    const res = await fetch(`${BASE_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        query: query.trim(),
        hybrid,
        hyde,
        top_k,
      }),
      signal: t.signal,
    });
    if (!res.ok) throw await parseError(res);
    return await res.json();
  } finally {
    t.clear();
  }
}

export async function uploadPdf(file, onProgress) {
  if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
    throw new Error("Only PDF files are accepted.");
  }

  const body = new FormData();
  body.append("file", file);

  return await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE_URL}/upload`);
    xhr.timeout = 30 * 60 * 1000;

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress?.(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onerror = () => reject(new Error("Could not reach the RAG backend."));
    xhr.ontimeout = () => reject(new Error("Upload timed out."));
    xhr.onload = () => {
      let data;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        data = xhr.responseText;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        return reject(new Error(data?.detail || "Upload failed."));
      }
      resolve(data);
    };
    xhr.send(body);
  });
}

export { BASE_URL };
