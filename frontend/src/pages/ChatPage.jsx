import { useEffect, useState } from "react";
import QueryPanel from "../components/QueryPanel";
import UploadPanel from "../components/UploadPanel";
import ResultsPanel from "../components/ResultsPanel";
import MetricsPanel from "../components/MetricsPanel";
import { queryRag, uploadPdf } from "../api/client";

const UPLOAD_STEPS_COUNT = 5;

export default function ChatPage({ onHome }) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hybrid, setHybrid] = useState(true);
  const [hyde, setHyde] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadPhase, setUploadPhase] = useState(null);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    if (uploadPhase !== "working") return;
    setStepIndex(0);
    const id = setInterval(() => {
      setStepIndex((i) => (i + 1) % UPLOAD_STEPS_COUNT);
    }, 3000);
    return () => clearInterval(id);
  }, [uploadPhase]);

  const submit = async (text) => {
    setMessages([{ role: "user", content: text }]);
    setMetrics(null);
    setLoading(true);
    try {
      const data = await queryRag({ query: text, hybrid, hyde });
      setMessages((m) => [...m, { role: "assistant", content: data.answer }]);
      setMetrics(data);
    } catch (e) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Unable to retrieve an answer. ${e.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const upload = async (file) => {
    setUploadBusy(true);
    setUploadPhase("working");
    try {
      await uploadPdf(file);
      setUploadPhase("done");
      setTimeout(() => setUploadPhase(null), 2400);
    } catch {
      setUploadPhase("error");
      setTimeout(() => setUploadPhase(null), 2600);
    } finally {
      setUploadBusy(false);
    }
  };

  const setMethod = (key, value) => {
    if (key === "hybrid" && !value && !hyde) return;
    if (key === "hyde" && !value && !hybrid) return;
    if (key === "hybrid") setHybrid(value);
    else setHyde(value);
  };

  return (
    <main className="app chat-page">
      <div className="orb" />
      <header>
        <button className="brand-link" type="button" onClick={onHome}>
          <div className="brand-mark">
            <img src="/icon.png" alt="" />
          </div>
          <div className="brand">
            <span>app</span>
          </div>
        </button>
        <div className="header-status">
          <i /> LOCAL KNOWLEDGE ENGINE
        </div>
      </header>

      <section className="chat-shell">
        <div className="chat-top">
          <span>CONVERSATION</span>
          <span className="live-dot">● ONLINE</span>
        </div>
        <ResultsPanel messages={messages} loading={loading} />
        <MetricsPanel data={metrics} />
        <QueryPanel
          onSubmit={submit}
          loading={loading}
          hybrid={hybrid}
          hyde={hyde}
          setMethod={setMethod}
        />
      </section>

      <UploadPanel
        onUpload={upload}
        busy={uploadBusy}
        phase={uploadPhase}
        stepIndex={stepIndex}
      />

      <footer>
        RAG APP <span>•</span> Built for local retrieval
      </footer>
    </main>
  );
}
