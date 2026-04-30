import { useCallback, useEffect, useState } from "react";
import type { InferResult } from "./api/client";
import { deleteResult, fetchResults, runInference } from "./api/client";
import { HistoryPanel } from "./components/HistoryPanel";
import { UploadPanel } from "./components/UploadPanel";
import { ResultViewer } from "./viewer/ResultViewer";
import "./App.css";

type AppState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "result"; result: InferResult }
  | { phase: "error"; message: string };

export default function App() {
  const [state, setState] = useState<AppState>({ phase: "idle" });
  const [threshold, setThreshold] = useState(0.5);
  const [history, setHistory] = useState<InferResult[]>([]);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory(await fetchResults());
    } catch {
      setHistory([]);
    }
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  const handleFile = useCallback(
    async (file: File) => {
      setState({ phase: "loading" });
      try {
        const result = await runInference(file, threshold);
        setState({ phase: "result", result });
        await refreshHistory();
      } catch (err) {
        setState({ phase: "error", message: err instanceof Error ? err.message : String(err) });
      }
    },
    [refreshHistory, threshold]
  );

  const reset = () => setState({ phase: "idle" });

  const selectHistory = (result: InferResult) => {
    setState({ phase: "result", result });
  };

  const removeHistory = async (id: string) => {
    try {
      await deleteResult(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
      if (state.phase === "result" && state.result.id === id) {
        setState({ phase: "idle" });
      }
    } catch (err) {
      setState({ phase: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">RibSeg Viewer</h1>
        <div className="header-controls">
          <label className="control-label">Threshold</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="threshold-input"
          />
          {state.phase !== "idle" && (
            <button className="reset-btn" onClick={reset}>
              New image
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        <HistoryPanel
          results={history}
          selectedId={state.phase === "result" ? state.result.id : undefined}
          onSelect={selectHistory}
          onDelete={removeHistory}
        />
        <section className="workspace">
          {state.phase === "idle" && (
            <UploadPanel onFile={handleFile} disabled={false} />
          )}

          {state.phase === "loading" && (
            <div className="status-message">
              <div className="spinner" />
              <span>Running inference...</span>
            </div>
          )}

          {state.phase === "error" && (
            <div className="error-panel">
              <div className="error-title">Inference failed</div>
              <div className="error-body">{state.message}</div>
              <button className="reset-btn" onClick={reset}>Try again</button>
            </div>
          )}

          {state.phase === "result" && (
            <ResultViewer result={state.result} />
          )}
        </section>
      </main>
    </div>
  );
}
