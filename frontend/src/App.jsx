import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef(null);

  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => () => clearInterval(timer.current), []);

  async function analyze() {
    setBusy(true);
    setError("");
    setResult(null);
    setElapsed(0);
    const started = Date.now();
    timer.current = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 500);
    try {
      const body = new FormData();
      body.append("image", file);
      const res = await fetch(`${API}/predict`, { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setResult(data);
    } catch (e) {
      setError(e.message === "Failed to fetch"
        ? "Cannot reach the API. Start the backend with: python -m backend.app"
        : e.message);
    } finally {
      clearInterval(timer.current);
      setBusy(false);
    }
  }

  const pct = (v) => `${(v * 100).toFixed(1)}%`;

  return (
    <main className="page">
      <header>
        <h1>MRI Review Assistant</h1>
        <p className="lede">
          Upload a brain MRI slice. The model names the likely class, shows where it looked,
          and says when it is too unsure to answer.
        </p>
      </header>

      <section className="panel">
        <label className="drop">
          <input type="file" accept="image/*" onChange={(e) => { setFile(e.target.files[0] || null); setResult(null); setError(""); }} />
          <span>{file ? file.name : "Choose an MRI image (JPG or PNG)"}</span>
        </label>
        <button onClick={analyze} disabled={!file || busy}>
          {busy ? `Analyzing… ${elapsed}s` : "Analyze image"}
        </button>
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {(preview || result) && (
        <section className="images">
          {preview && (
            <figure>
              <img src={preview} alt="Uploaded MRI" />
              <figcaption>Your image</figcaption>
            </figure>
          )}
          {result && (
            <figure>
              <img src={`data:image/png;base64,${result.heatmap}`} alt="Grad-CAM heatmap" />
              <figcaption>Where the model looked (Grad-CAM)</figcaption>
            </figure>
          )}
        </section>
      )}

      {result && (
        <section className={`result ${result.needs_review ? "review" : "ok"}`}>
          {result.needs_review ? (
            <>
              <h2>Low confidence: needs expert review</h2>
              <p>The model's repeated passes disagree too much. Do not rely on this result.</p>
              <p className="muted">Its best guess was {result.label} ({pct(result.confidence)}).</p>
            </>
          ) : (
            <>
              <h2>{result.label}</h2>
              <p>Confidence {pct(result.confidence)}</p>
            </>
          )}

          <div className="meter" aria-label="Uncertainty">
            <div className="meter-track">
              <div className="meter-fill" style={{ width: `${Math.min(100, result.uncertainty * 100)}%` }} />
              <div className="meter-limit" style={{ left: `${result.threshold * 100}%` }} title="Review threshold" />
            </div>
            <span>Uncertainty {result.uncertainty.toFixed(2)} (review above {result.threshold.toFixed(2)})</span>
          </div>

          <ul className="probs">
            {Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([name, p]) => (
                <li key={name}><span>{name}</span><span>{pct(p)}</span></li>
              ))}
          </ul>
          {result.seconds !== undefined && (
            <p className="muted took">Prediction took {result.seconds} s</p>
          )}
        </section>
      )}

      <footer>
        Educational project. Not a medical device and not for diagnosis.
      </footer>
    </main>
  );
}
