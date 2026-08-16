import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";

const LOADING_STEPS = [
  "Embedding query",
  "Retrieving documents",
  "Cross encoder reranking",
  "Generating response",
];

export default function ResultsPanel({ messages, loading }) {
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    if (!loading) {
      setStepIndex(0);
      return;
    }
    setStepIndex(0);
    const id = setInterval(() => {
      setStepIndex((i) => (i + 1) % LOADING_STEPS.length);
    }, 4000);
    return () => clearInterval(id);
  }, [loading]);

  return (
    <div className="messages">
      {messages.length === 0 && !loading && (
        <div className="empty">
          <p>Your retrieval workspace is ready.</p>
        </div>
      )}
      {messages.map((m, i) => (
        <div key={i} className={`message ${m.role}`}>
          <span className="message-label">
            {m.role === "user" ? "YOU" : "RAG APP"}
          </span>
          {m.role === "assistant" && !m.error ? (
            <div className="bubble markdown-body">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeSanitize]}
              >
                {m.content}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="bubble">{m.content}</div>
          )}
        </div>
      ))}
      {loading && (
        <div className="loading-state">
          <div className="astronaut">◉</div>
          <span className="loading-status-wrap">
            <b key={stepIndex} className="loading-status">
              {LOADING_STEPS[stepIndex]}
            </b>
          </span>
        </div>
      )}
    </div>
  );
}
