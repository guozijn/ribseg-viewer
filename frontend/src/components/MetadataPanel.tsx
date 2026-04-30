import type { InferResult } from "../api/client";

interface Props {
  result: InferResult;
}

export function MetadataPanel({ result }: Props) {
  return (
    <div className="metadata-panel">
      <div className="meta-row">
        <span className="meta-key">Result ID</span>
        <span className="meta-value meta-mono">{result.id.slice(0, 8)}...</span>
      </div>
      <div className="meta-row">
        <span className="meta-key">Image size</span>
        <span className="meta-value">{result.original_width} x {result.original_height}</span>
      </div>
      <div className="meta-row">
        <span className="meta-key">Threshold</span>
        <span className="meta-value">{result.threshold.toFixed(2)}</span>
      </div>
      <div className="meta-row">
        <span className="meta-key">Inference time</span>
        <span className="meta-value">{(result.inference_time_s * 1000).toFixed(0)} ms</span>
      </div>
    </div>
  );
}
