import { useEffect, useState } from "react";

export default function MetricsPanel({ data, status }) {
  const [shown, setShown] = useState(null);

  useEffect(() => {
    if (data) setShown(data);
  }, [data]);

  const retrieved = shown?.documents_retrieved ?? "—";
  const used = shown?.documents_used ?? "—";
  const methods = shown?.methods?.join(" + ") || "—";
  const hasMetrics = Boolean(data);
  const hasStatus = status != null && status !== "";
  const isError = typeof status === "number" && (status < 200 || status >= 300);

  return (
    <div
      className={[
        "metrics",
        hasMetrics || hasStatus ? "has-data" : "",
        isError ? "is-error" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="metrics-text">
        {hasMetrics ? (
          <>
            <span>
              <b>{retrieved}</b> documents retrieved
            </span>
            <span>
              <b>{used}</b> documents used
            </span>
            <span>
              <b>{methods}</b> logic
            </span>
          </>
        ) : (
          <span className="metrics-placeholder">
            {isError ? "Request failed" : ""}
          </span>
        )}
      </div>
      {hasStatus && (
        <div className="metrics-status" aria-label={`HTTP status ${status}`}>
          <span className="metrics-status-label">status</span>
          <b>{status}</b>
        </div>
      )}
    </div>
  );
}
