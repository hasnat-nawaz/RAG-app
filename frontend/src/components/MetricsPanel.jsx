import { useEffect, useState } from "react";

export default function MetricsPanel({ data }) {
  const [shown, setShown] = useState(null);

  useEffect(() => {
    if (data) setShown(data);
  }, [data]);

  const retrieved = shown?.documents_retrieved ?? "—";
  const used = shown?.documents_used ?? "—";
  const methods = shown?.methods?.join(" + ") || "—";

  return (
    <div className={`metrics${data ? " has-data" : ""}`}>
      <div className="metrics-text">
        <span>
          <b>{retrieved}</b> documents retrieved
        </span>
        <span>
          <b>{used}</b> documents used
        </span>
        <span>
          <b>{methods}</b> logic
        </span>
      </div>
    </div>
  );
}
