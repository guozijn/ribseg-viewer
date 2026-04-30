import type { InferResult } from "../api/client";

interface Props {
  results: InferResult[];
  selectedId?: string;
  onSelect: (result: InferResult) => void;
  onDelete: (id: string) => void;
}

function formatTime(value?: string): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

export function HistoryPanel({ results, selectedId, onSelect, onDelete }: Props) {
  return (
    <aside className="history-panel">
      <div className="history-header">
        <span>History</span>
        <span className="history-count">{results.length}</span>
      </div>
      <div className="history-list">
        {results.length === 0 && (
          <div className="history-empty">No saved results</div>
        )}
        {results.map((result) => (
          <div
            key={result.id}
            className={`history-item${result.id === selectedId ? " active" : ""}`}
          >
            <button
              className="history-select"
              type="button"
              onClick={() => onSelect(result)}
            >
              <span className="history-name">
                {result.filename || result.id.slice(0, 8)}
              </span>
              <span className="history-meta">
                {result.original_width} x {result.original_height}
              </span>
              <span className="history-meta">{formatTime(result.created_at)}</span>
            </button>
            <button
              className="history-delete"
              type="button"
              onClick={() => onDelete(result.id)}
              title="Delete result"
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
