import { useEffect, useRef, useState } from "react";
import { estimateUploadNotice, noticeText } from "../utils/uploadEstimate";

const UPLOAD_STEPS = [
  "Processing document",
  "Converting to markdown",
  "Chunking",
  "Embedding",
  "Saving to vector database",
];

const EXPAND_MS = 960;
const STATUS_FADE_MS = 750;
const COLLAPSE_MS = 960;
const NOTICE_MS = 4000;
const STEP_MS = 4000; // 1.5s in + 1s hold + 1.5s out

export default function UploadPanel({ onUpload, busy, phase }) {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [drag, setDrag] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [showStatus, setShowStatus] = useState(false);
  const [showIdle, setShowIdle] = useState(true);
  const [statusText, setStatusText] = useState("");
  const [statusKind, setStatusKind] = useState("working");
  const [statusKey, setStatusKey] = useState("idle");
  const ref = useRef();
  const timers = useRef([]);
  const phaseRef = useRef(phase);
  const noticeKindRef = useRef(null);

  const clearTimers = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };

  const later = (fn, ms) => {
    const id = setTimeout(fn, ms);
    timers.current.push(id);
    return id;
  };

  const startStepCycle = () => {
    let index = 0;
    setStatusKind("working");
    setStatusText(UPLOAD_STEPS[0]);
    setStatusKey("step-0");
    setShowStatus(true);
    const id = setInterval(() => {
      index = (index + 1) % UPLOAD_STEPS.length;
      setStatusText(UPLOAD_STEPS[index]);
      setStatusKey(`step-${index}`);
    }, STEP_MS);
    timers.current.push(id);
  };

  useEffect(() => {
    const previous = phaseRef.current;
    phaseRef.current = phase;
    clearTimers();

    if (phase === "working") {
      setShowIdle(false);
      setShowStatus(false);
      setExpanded(true);
      const notice = noticeKindRef.current;
      later(() => {
        if (notice) {
          setStatusKind("notice");
          setStatusText(noticeText(notice));
          setStatusKey(`notice-${notice}`);
          setShowStatus(true);
          later(() => {
            setShowStatus(false);
            later(() => startStepCycle(), STATUS_FADE_MS);
          }, NOTICE_MS);
        } else {
          startStepCycle();
        }
      }, EXPAND_MS);
      return clearTimers;
    }

    if (phase === "done" || phase === "error") {
      setExpanded(true);
      setShowIdle(false);
      setStatusKind(phase);
      setStatusText(
        phase === "done" ? "Source added successfully" : "Upload failed",
      );
      setStatusKey(phase);
      setShowStatus(true);
      return clearTimers;
    }

    if (previous == null && phase == null) {
      return clearTimers;
    }

    setShowStatus(false);
    later(() => {
      setExpanded(false);
      later(() => {
        setStatusText("");
        setStatusKind("working");
        setShowIdle(true);
        noticeKindRef.current = null;
      }, COLLAPSE_MS);
    }, STATUS_FADE_MS);

    return clearTimers;
  }, [phase]);

  const choose = (f) => {
    if (f && !f.name.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF files are accepted.");
      return;
    }
    setFile(f);
  };

  const close = () => {
    setOpen(false);
    setFile(null);
    setDrag(false);
  };

  const startUpload = async () => {
    if (!file || busy) return;
    const selected = file;
    close();
    try {
      noticeKindRef.current = await estimateUploadNotice(selected);
    } catch {
      noticeKindRef.current = null;
    }
    onUpload(selected);
  };

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (e) => {
      if (e.key !== "Enter" || e.nativeEvent?.isComposing) return;
      if (!file || busy) return;
      e.preventDefault();
      startUpload();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, file, busy]);

  const dockBusy = Boolean(phase) || busy;
  const showPulseRing =
    showStatus && (statusKind === "working" || statusKind === "notice");

  return (
    <>
      <div
        className={[
          "upload-dock",
          expanded ? "expanded" : "",
          showStatus ? "status-visible" : "",
          showIdle ? "idle-visible" : "",
          statusKind === "done" ? "done" : "",
          statusKind === "error" ? "error" : "",
          statusKind === "notice" ? "notice" : "",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {showPulseRing ? (
          <span
            key={statusKey}
            className={
              statusKind === "notice"
                ? "upload-dock-pulse-ring notice-ring"
                : "upload-dock-pulse-ring"
            }
            aria-hidden="true"
          />
        ) : null}
        <button
          className="upload-dock-hit"
          type="button"
          disabled={dockBusy}
          onClick={() => {
            if (!dockBusy) setOpen(true);
          }}
          aria-label="Add source"
        >
          <span className="upload-dock-idle">＋ Add source</span>
          <span className="upload-dock-status" aria-live="polite">
            {statusText ? (
              <b
                key={statusKey}
                className={
                  statusKind === "working" || statusKind === "notice"
                    ? "process-status"
                    : "process-result"
                }
              >
                {statusText}
              </b>
            ) : null}
          </span>
        </button>
      </div>

      {open && (
        <div className="modal-backdrop" onClick={close} role="presentation">
          <div
            className="upload-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <button className="close" type="button" onClick={close}>
              ×
            </button>
            <div className="eyebrow">DOCUMENT PIPELINE</div>
            <h2>Bring a new source into orbit.</h2>
            <p>
              This document will be saved locally, converted to clean markdown,
              chunked, and stored as embeddings in the local vector database.
            </p>
            <div
              className={`dropzone ${drag ? "drag" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDrag(false);
                choose(e.dataTransfer.files[0]);
              }}
              onClick={() => ref.current.click()}
            >
              <input
                ref={ref}
                type="file"
                accept="application/pdf,.pdf"
                hidden
                onChange={(e) => choose(e.target.files[0])}
              />
              <strong>{file ? `📄 ${file.name}` : "Drop a PDF here"}</strong>
              <span>or choose from device · PDF only</span>
            </div>
            {file && (
              <div className="file-line">
                <span>{file.name}</span>
                <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}
            <button
              className="primary"
              type="button"
              disabled={!file || busy}
              onClick={startUpload}
            >
              Upload to pipeline
            </button>
          </div>
        </div>
      )}
    </>
  );
}
