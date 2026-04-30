import { useState } from "react";
import type { InferResult } from "../api/client";
import {
  resultImageUrl,
  resultLabelOverlayUrl,
  resultLabelMaskUrl,
  resultMaskUrl,
  resultOverlayUrl,
} from "../api/client";
import { LabelToggles } from "../components/LabelToggles";
import { MetadataPanel } from "../components/MetadataPanel";
import { OverlayCanvas } from "./OverlayCanvas";

interface Props {
  result: InferResult;
}

export function ResultViewer({ result }: Props) {
  const [opacity, setOpacity] = useState(0.6);
  const [visible, setVisible] = useState<Set<string>>(new Set(result.labels));

  const toggle = (label: string) => {
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else next.add(label);
      return next;
    });
  };

  const toggleAll = (side: "R" | "L") => {
    const sideLabels = result.labels.filter((l) => l.startsWith(side));
    const allVisible = sideLabels.every((l) => visible.has(l));
    setVisible((prev) => {
      const next = new Set(prev);
      if (allVisible) sideLabels.forEach((l) => next.delete(l));
      else sideLabels.forEach((l) => next.add(l));
      return next;
    });
  };

  const imageUrl = resultImageUrl(result.id);
  const overlayUrl = resultOverlayUrl(result.id);

  return (
    <div className="result-viewer">
      <div className="viewer-main">
        <OverlayCanvas
          imageUrl={imageUrl}
          labels={result.labels}
          visible={visible}
          getOverlayUrl={(label) => resultLabelOverlayUrl(result.id, label)}
          opacity={opacity}
        />

        <div className="viewer-controls">
          <div className="control-row">
            <label className="control-label">Overlay opacity</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="opacity-slider"
            />
            <span className="control-value">{Math.round(opacity * 100)}%</span>
          </div>

          <div className="download-row">
            <a href={overlayUrl} download={`overlay-${result.id}.png`} className="dl-btn">
              Download overlay
            </a>
            <a href={resultMaskUrl(result.id)} download={`mask-${result.id}.png`} className="dl-btn">
              Download mask
            </a>
          </div>
        </div>
      </div>

      <div className="viewer-sidebar">
        <MetadataPanel result={result} />
        <LabelToggles
          labels={result.labels}
          visible={visible}
          areas={result.areas}
          onToggle={toggle}
          onToggleAll={toggleAll}
        />
        <div className="per-rib-downloads">
          <div className="sidebar-section-title">Per-rib masks</div>
          {result.labels.map((label) => (
            <a
              key={label}
              href={resultLabelMaskUrl(result.id, label)}
              download={`mask-${label}-${result.id}.png`}
              className="rib-dl-link"
            >
              {label}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
