interface Props {
  labels: string[];
  visible: Set<string>;
  areas: Record<string, number>;
  onToggle: (label: string) => void;
  onToggleAll: (side: "R" | "L") => void;
}

export function LabelToggles({ labels, visible, areas, onToggle, onToggleAll }: Props) {
  const right = labels.filter((l) => l.startsWith("R"));
  const left = labels.filter((l) => l.startsWith("L"));

  const renderSide = (side: "R" | "L", ribs: string[]) => (
    <div className="label-side">
      <div className="label-side-header">
        <span>{side === "R" ? "Right" : "Left"}</span>
        <button className="toggle-all-btn" onClick={() => onToggleAll(side)}>
          Toggle all
        </button>
      </div>
      {ribs.map((label) => (
        <label key={label} className={`rib-toggle${visible.has(label) ? " active" : ""}`}>
          <input
            type="checkbox"
            checked={visible.has(label)}
            onChange={() => onToggle(label)}
          />
          <span className="rib-label-name">{label}</span>
          <span className="rib-area">{areas[label] ?? 0} px</span>
        </label>
      ))}
    </div>
  );

  return (
    <div className="label-toggles">
      {renderSide("R", right)}
      {renderSide("L", left)}
    </div>
  );
}
