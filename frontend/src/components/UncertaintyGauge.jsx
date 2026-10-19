/** Half-circle gauge: the arc fills with uncertainty, the tick marks the review threshold. */
export default function UncertaintyGauge({ value, threshold, review }) {
  const v = Math.min(1, Math.max(0, value));
  const t = Math.min(1, Math.max(0, threshold));
  const angle = Math.PI * (1 - t);
  const x1 = 100 + 68 * Math.cos(angle);
  const y1 = 100 - 68 * Math.sin(angle);
  const x2 = 100 + 94 * Math.cos(angle);
  const y2 = 100 - 94 * Math.sin(angle);
  const arc = "M20 100 A80 80 0 0 1 180 100";

  return (
    <figure className="gauge">
      <svg
        viewBox="0 0 200 124"
        role="img"
        aria-label={`Uncertainty ${value.toFixed(2)}. The model asks for review above ${threshold.toFixed(2)}.`}
      >
        <path d={arc} pathLength="100" className="gauge-track" />
        <path
          d={arc}
          pathLength="100"
          className={`gauge-fill ${review ? "review" : "ok"}`}
          strokeDasharray={`${v * 100} 100`}
        />
        <line x1={x1} y1={y1} x2={x2} y2={y2} className="gauge-limit" />
        <text x="100" y="94" textAnchor="middle" className="gauge-num">{value.toFixed(2)}</text>
        <text x="100" y="116" textAnchor="middle" className="gauge-cap">uncertainty</text>
      </svg>
      <figcaption>The tick marks the review limit ({threshold.toFixed(2)}).</figcaption>
    </figure>
  );
}
