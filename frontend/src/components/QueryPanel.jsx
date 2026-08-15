import { useEffect, useRef, useState } from "react";
import MethodSelector from "./MethodSelector";

export default function QueryPanel({
  onSubmit,
  loading,
  hybrid,
  hyde,
  setMethod,
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef(null);

  const resize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  useEffect(() => {
    resize();
  }, [value]);

  const send = () => {
    if (value.trim() && !loading && (hybrid || hyde)) {
      onSubmit(value);
      setValue("");
    }
  };

  return (
    <div className="query-area">
      <div className="query-input-shell">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (
              e.key === "Enter" &&
              !e.shiftKey &&
              !e.nativeEvent.isComposing &&
              e.keyCode !== 229
            ) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask questions from your documents…"
          aria-label="Question"
        />
      </div>
      <div className="query-controls">
        <MethodSelector hybrid={hybrid} hyde={hyde} onChange={setMethod} />
        <button
          className="send"
          onClick={send}
          type="button"
          disabled={!value.trim() || loading || (!hybrid && !hyde)}
        >
          {loading ? "…" : "↑"}
        </button>
      </div>
    </div>
  );
}
