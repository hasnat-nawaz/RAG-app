const PAGES_PER_CHUNK = 4;
const MAX_BATCH_REQUESTS = 15;

export async function estimatePdfPages(file) {
  try {
    const buf = await file.arrayBuffer();
    const text = new TextDecoder("latin1").decode(buf);
    const matches = text.match(/\/Type\s*\/Page(?!\s*s)/g);
    return Math.max(1, matches?.length || 1);
  } catch {
    return Math.max(1, Math.round(file.size / 80000));
  }
}

export async function estimateUploadNotice(file) {
  const pages = await estimatePdfPages(file);
  const chunks = Math.ceil(pages / PAGES_PER_CHUNK);
  const batches = Math.ceil(chunks / MAX_BATCH_REQUESTS);
  if (batches > 1) return "minutes";
  if (chunks > 1) return "moment";
  return null;
}

export function noticeText(kind) {
  if (kind === "minutes") return "This may take a couple of minutes.";
  if (kind === "moment") return "This may take a moment.";
  return null;
}
