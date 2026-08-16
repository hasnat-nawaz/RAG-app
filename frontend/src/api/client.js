const BASE_URL = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function pickMessage(value) {
  if (value == null || value === "") return null;
  if (typeof value === "string") return value.trim() || null;
  if (typeof value !== "object") return String(value);

  const nested =
    value.message ||
    value.error?.message ||
    value.detail?.message ||
    (typeof value.detail === "string" ? value.detail : null);
  if (nested) return pickMessage(nested);

  if (Array.isArray(value.detail)) {
    const parts = value.detail
      .map((item) => pickMessage(item?.msg || item?.message || item))
      .filter(Boolean);
    if (parts.length) return parts.join(" ");
  }

  return null;
}

async function parseError(response) {
  const raw = await response.text();
  let body = null;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    body = raw;
  }

  const message =
    pickMessage(body) ||
    pickMessage(body?.detail) ||
    (typeof raw === "string" && raw.trim()) ||
    response.statusText ||
    "Request failed.";

  return new ApiError(message, response.status);
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
    const data = await res.json();
    return { ...data, status: res.status };
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err?.name === "AbortError") {
      throw new ApiError("The request timed out. Please try again.", 408);
    }
    throw new ApiError(
      err?.message || "Could not reach the RAG backend.",
      0,
    );
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
        const message =
          pickMessage(data) ||
          (typeof data === "string" ? data : null) ||
          "Upload failed.";
        return reject(new ApiError(message, xhr.status));
      }
      resolve(data);
    };
    xhr.send(body);
  });
}

export async function checkHealth({ timeoutMs = 2500 } = {}) {
  const t = withTimeout(timeoutMs);
  try {
    const res = await fetch(`${BASE_URL}/health`, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: t.signal,
      cache: "no-store",
    });
    if (!res.ok) return false;
    const data = await res.json().catch(() => null);
    return data?.status === "ok";
  } catch {
    return false;
  } finally {
    t.clear();
  }
}

export { BASE_URL };
