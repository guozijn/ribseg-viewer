import { useEffect, useRef, useState } from "react";

interface Props {
  imageUrl: string;
  labels: string[];
  visible: Set<string>;
  getOverlayUrl: (label: string) => string;
  opacity: number;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load image: ${src}`));
    img.src = src;
  });
}

export function OverlayCanvas({ imageUrl, labels, visible, getOverlayUrl, opacity }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageCacheRef = useRef<Map<string, HTMLImageElement>>(new Map());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let cancelled = false;

    const cachedImage = async (src: string) => {
      const cached = imageCacheRef.current.get(src);
      if (cached) return cached;
      const img = await loadImage(src);
      imageCacheRef.current.set(src, img);
      return img;
    };

    const draw = async () => {
      if (cancelled) return;
      const urls = [imageUrl, ...labels.filter((label) => visible.has(label)).map(getOverlayUrl)];
      const hasUncachedImage = urls.some((url) => !imageCacheRef.current.has(url));
      if (hasUncachedImage) setLoading(true);

      const baseImg = await cachedImage(imageUrl);
      const activeLabels = labels.filter((label) => visible.has(label));
      const overlayImages = await Promise.all(
        activeLabels.map((label) => cachedImage(getOverlayUrl(label)))
      );
      if (cancelled) return;

      const { width: cw, height: ch } = container.getBoundingClientRect();
      const naturalW = baseImg.naturalWidth;
      const naturalH = baseImg.naturalHeight;
      const scale = Math.min(cw / naturalW, ch / naturalH, 1);
      const drawW = Math.round(naturalW * scale);
      const drawH = Math.round(naturalH * scale);
      canvas.width = drawW;
      canvas.height = drawH;
      ctx.drawImage(baseImg, 0, 0, drawW, drawH);

      overlayImages.forEach((overlayImg) => {
        ctx.globalAlpha = opacity;
        ctx.drawImage(overlayImg, 0, 0, drawW, drawH);
      });
      ctx.globalAlpha = 1;
      setLoading(false);
    };

    draw().catch(() => {
      if (!cancelled) setLoading(false);
    });

    return () => { cancelled = true; };
  }, [imageUrl, labels, visible, getOverlayUrl, opacity]);

  return (
    <div ref={containerRef} className="canvas-container">
      {loading && (
        <div className="canvas-loading" aria-live="polite">
          <div className="spinner" />
          <span>Loading image...</span>
        </div>
      )}
      <canvas ref={canvasRef} className="overlay-canvas" />
    </div>
  );
}
