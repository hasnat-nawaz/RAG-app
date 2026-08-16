import { useEffect, useRef, useState } from "react";

export default function MethodSelector({ hybrid, hyde, onChange }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  return (
    <div className="method-wrap" ref={wrapRef}>
      <button
        className="method-trigger"
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        Retrieval logic{" "}
        <span className="method-chevron" aria-hidden="true">
          <svg viewBox="0 0 12 8" width="9" height="6" fill="none">
            <path
              d="M1.5 1.75L6 6.25L10.5 1.75"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>
      {open && (
        <div className="method-pop">
          <p>Select one or more retrieval steps.</p>
          <label className={`method-card ${hybrid ? "active" : ""}`}>
            <input
              type="checkbox"
              checked={hybrid}
              onChange={(e) => onChange("hybrid", e.target.checked)}
            />
            <span>
              <b>Hybrid search</b>
              <small>Semantic search + BM25</small>
            </span>
            {hybrid && <i>SELECTED</i>}
          </label>
          <label className={`method-card ${hyde ? "active" : ""}`}>
            <input
              type="checkbox"
              checked={hyde}
              onChange={(e) => onChange("hyde", e.target.checked)}
            />
            <span>
              <b>HyDE</b>
              <small>Hypothetical document expansion</small>
            </span>
            {hyde && <i>SELECTED</i>}
          </label>
        </div>
      )}
    </div>
  );
}
