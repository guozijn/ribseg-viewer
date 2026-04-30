import React, { useCallback, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled: boolean;
}

export function UploadPanel({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!file.type.startsWith("image/")) return;
      onFile(file);
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  return (
    <div
      className={`upload-zone${dragOver ? " drag-over" : ""}${disabled ? " disabled" : ""}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        style={{ display: "none" }}
        onChange={(e) => handleFiles(e.target.files)}
        disabled={disabled}
      />
      <div className="upload-label">
        {disabled ? "Running inference..." : "Drop a chest X-ray here or click to select"}
      </div>
      <div className="upload-hint">PNG, JPEG, or WebP</div>
    </div>
  );
}
