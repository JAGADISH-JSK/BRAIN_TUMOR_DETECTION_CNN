/** One row per class, sorted from most to least likely. */
export default function ProbabilityBars({ probabilities }) {
  const rows = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <ul className="probs">
      {rows.map(([name, p], i) => (
        <li key={name} className={i === 0 ? "lead" : undefined}>
          <span className="p-name">{name}</span>
          <span className="p-bar" aria-hidden="true"><span style={{ width: `${p * 100}%` }} /></span>
          <span className="p-val">{(p * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}
