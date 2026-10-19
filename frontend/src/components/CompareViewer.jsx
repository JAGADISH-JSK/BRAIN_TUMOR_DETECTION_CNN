import { useState } from "react";

/**
 * The image stage. Shows the uploaded slice and, once a result exists, the Grad-CAM overlay.
 * view: "original" | "heatmap" | "split". In split view a native range input sits on top of the
 * stage, so the divider works with mouse, touch and keyboard (arrow keys).
 */
export default function CompareViewer({ original, heatmap, view }) {
  const [pos, setPos] = useState(50);
  const split = Boolean(heatmap) && view === "split";
  const showHeat = Boolean(heatmap) && view !== "original";

  return (
    <div className="stage">
      <img className="layer" src={original} alt="Uploaded MRI slice" draggable="false" />
      {showHeat && (
        <img
          className="layer"
          src={heatmap}
          alt="Grad-CAM overlay showing where the model looked"
          draggable="false"
          style={split ? { clipPath: `inset(0 ${100 - pos}% 0 0)` } : undefined}
        />
      )}

      {split && <span className="divider" style={{ left: `${pos}%` }} aria-hidden="true" />}
      {heatmap && (
        <>
          {(view === "original" || split) && <span className="chip chip-left">Original</span>}
          {(view === "heatmap" || split) && <span className="chip chip-right">Grad-CAM</span>}
        </>
      )}

      {split && (
        <input
          className="split-range"
          type="range"
          min="0"
          max="100"
          step="1"
          value={pos}
          onChange={(e) => setPos(Number(e.target.value))}
          aria-label="Drag to compare the original image with the Grad-CAM overlay"
        />
      )}
    </div>
  );
}
