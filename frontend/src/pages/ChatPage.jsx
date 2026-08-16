import { useEffect, useState } from "react";
import QueryPanel from "../components/QueryPanel";
import UploadPanel from "../components/UploadPanel";
import ResultsPanel from "../components/ResultsPanel";
import MetricsPanel from "../components/MetricsPanel";
import { checkHealth, queryRag, uploadPdf } from "../api/client";

const OFFLINE_POLL_MS = 4000;

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [hybrid, setHybrid] = useState(true);
  const [hyde, setHyde] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [statusCode, setStatusCode] = useState(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadPhase, setUploadPhase] = useState(null);
  const [online, setOnline] = useState(true);

  useEffect(() => {
    if (online) return undefined;

    let cancelled = false;

    const poll = async () => {
      if (typeof navigator !== "undefined" && !navigator.onLine) return;
      const ok = await checkHealth();
      if (!cancelled && ok) setOnline(true);
    };

    poll();
    const id = setInterval(poll, OFFLINE_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [online]);

  const submit = async (text) => {
    setMessages([{ role: "user", content: text }]);
    setMetrics(null);
    setStatusCode(null);

    const browserOffline =
      typeof navigator !== "undefined" && !navigator.onLine;
    const reachable = browserOffline ? false : await checkHealth();

    if (!reachable) {
      setOnline(false);
      setStatusCode(0);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          error: true,
          content: "Server is offline.",
        },
      ]);
      return;
    }

    setOnline(true);
    setLoading(true);
    try {
      const data = await queryRag({ query: text, hybrid, hyde });
      setMessages((m) => [...m, { role: "assistant", content: data.answer }]);
      setMetrics(data);
      setStatusCode(data.status ?? 200);
      setOnline(true);
    } catch (e) {
      const code = typeof e.status === "number" ? e.status : 500;
      const unreachable = code === 0;
      if (unreachable) setOnline(false);
      else setOnline(true);
      const message = unreachable
        ? "Server is offline."
        : e.message?.trim() ||
          "Something went wrong while retrieving an answer. Please try again.";
      setStatusCode(code || 500);
      setMetrics(null);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          error: true,
          content: message,
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
      setOnline(true);
      setTimeout(() => setUploadPhase(null), 5800);
    } catch (e) {
      if (e?.status === 0) setOnline(false);
      setUploadPhase("error");
      setTimeout(() => setUploadPhase(null), 3200);
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
    <div className="page chat-page-content" key="chat">
      <section className="chat-shell">
        <div className="chat-top">
          <span>CONVERSATION</span>
          <span className={`live-status ${online ? "is-online" : "is-offline"}`}>
            <i className="live-status-dot" aria-hidden="true" />
            {online ? "ONLINE" : "OFFLINE"}
          </span>
        </div>
        <ResultsPanel messages={messages} loading={loading} />
        <MetricsPanel data={metrics} status={statusCode} />
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
      />
    </div>
  );
}
