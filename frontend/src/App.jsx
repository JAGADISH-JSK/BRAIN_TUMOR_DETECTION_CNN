import { useEffect, useRef, useState } from "react";
import CompareViewer from "./components/CompareViewer.jsx";
import UncertaintyGauge from "./components/UncertaintyGauge.jsx";
import ProbabilityBars from "./components/ProbabilityBars.jsx";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";
const MAX_HISTORY = 8;

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const fileSize = (b) => (b > 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);

const STATUS_TEXT = {
  checking: "Checking the server",
  ready: "Model ready",
  "no-model": "Server is up, but no model is loaded",
  offline: "Server offline. Run: python -m backend.app",
};

/** Asks the backend whether it is up and has a model, and keeps checking. */
function useBackendStatus() {
  const [status, setStatus] = useState("checking");
  useEffect(() => {
    let stopped = false;
    let timer;
    async function check() {
      let next = "offline";
      try {
        const res = await fetch(`${API}/health`);
        const data = await res.json();
        next = data.model_loaded ? "ready" : "no-model";
      } catch {
        next = "offline";
      }
      if (stopped) return;
      setStatus(next);
      timer = setTimeout(check, next === "ready" ? 30000 : 4000);
    }
    check();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);
  return status;
}

function UploadIcon() {
  return (
    <svg width="44" height="44" viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <ellipse cx="24" cy="22" rx="14" ry="16" stroke="currentColor" strokeWidth="2" />
      <path d="M17 24c2-5 5-7 7-7s5 2 7 7M24 17v-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M24 44V32m0 0-5 5m5-5 5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function App() {
  const status = useBackendStatus();
  const [image, setImage] = useState(null); // { file, name, size, url }
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [view, setView] = useState("split");
  const [history, setHistory] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(false);
  const timer = useRef(null);
  const urls = useRef([]);

  useEffect(() => () => {
    clearInterval(timer.current);
    urls.current.forEach((u) => URL.revokeObjectURL(u));
  }, []);

  function takeFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("That file is not an image. Choose a JPG or PNG MRI slice.");
      return;
    }
    const url = URL.createObjectURL(file);
    urls.current.push(url);
    setImage({ file, name: file.name, size: file.size, url });
    setResult(null);
    setError("");
  }

  async function analyze() {
    if (!image?.file || busy) return;
    setBusy(true);
    setError("");
    setResult(null);
    setElapsed(0);
    const started = Date.now();
    timer.current = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 500);
    try {
      const body = new FormData();
      body.append("image", image.file);
      const res = await fetch(`${API}/predict`, { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "The request failed.");
      setResult(data);
      setView("split");
      setHistory((h) => [{ id: Date.now(), name: image.name, url: image.url, result: data }, ...h].slice(0, MAX_HISTORY));
    } catch (e) {
      setError(
        e.message === "Failed to fetch"
          ? "Cannot reach the server. Start it with: python -m backend.app"
          : e.message
      );
    } finally {
      clearInterval(timer.current);
      setBusy(false);
    }
  }

  function openFromHistory(entry) {
    setImage({ file: null, name: entry.name, size: null, url: entry.url });
    setResult(entry.result);
    setError("");
    setView("split");
  }

  async function copySummary() {
    const r = result;
    const lines = [
      "MRI Review Assistant (educational project, not a medical device)",
      `File: ${image?.name ?? "unknown"}`,
      r.needs_review ? `Result: low confidence, needs expert review (best guess ${r.label}, ${pct(r.confidence)})` : `Result: ${r.label}, confidence ${pct(r.confidence)}`,
      `Uncertainty: ${r.uncertainty.toFixed(2)} (review above ${r.threshold.toFixed(2)})`,
      ...Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]).map(([n, p]) => `  ${n}: ${pct(p)}`),
    ];
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setError("Copy is blocked by the browser. Select the text on the page instead.");
    }
  }

  const hasImage = Boolean(image);

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>MRI Review Assistant</h1>
          <p className="lede">
            Upload a brain MRI slice. The model names the likely tumor type, shows where it looked,
            and tells you when it is too unsure to answer.
          </p>
        </div>
        <p className={`status status-${status}`} role="status">
          <span className="dot" aria-hidden="true" />
          {STATUS_TEXT[status]}
        </p>
      </header>

      <main className="workspace">
        <section className="viewer" aria-label="Image viewer">
          {hasImage ? (
            <CompareViewer original={image.url} heatmap={result ? `data:image/png;base64,${result.heatmap}` : null} view={view} />
          ) : (
            <label
              className={`drop ${dragging ? "over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); takeFile(e.dataTransfer.files[0]); }}
            >
              <input type="file" accept="image/*" onChange={(e) => takeFile(e.target.files[0])} />
              <UploadIcon />
              <strong>Drop an MRI slice here</strong>
              <span>or click to browse. JPG or PNG, up to 8 MB.</span>
            </label>
          )}

          {result && (
            <div className="views" role="group" aria-label="Choose what the viewer shows">
              {[["split", "Compare"], ["original", "Original"], ["heatmap", "Grad-CAM"]].map(([key, label]) => (
                <button key={key} type="button" aria-pressed={view === key} onClick={() => setView(key)}>
                  {label}
                </button>
              ))}
            </div>
          )}
        </section>

        <aside className="side">
          <section className="panel scan">
            <h2>Scan</h2>
            {hasImage ? (
              <p className="file">
                <span className="file-name">{image.name}</span>
                {image.size ? <span className="file-size">{fileSize(image.size)}</span> : null}
              </p>
            ) : (
              <p className="muted">No image selected yet.</p>
            )}
            <div className="actions">
              <button type="button" className="primary" onClick={analyze} disabled={!image?.file || busy || status !== "ready"}>
                {busy ? `Analyzing… ${elapsed}s` : "Analyze image"}
              </button>
              <label className="button secondary">
                <input type="file" accept="image/*" onChange={(e) => takeFile(e.target.files[0])} />
                {hasImage ? "Choose another" : "Browse files"}
              </label>
            </div>
            {busy && <p className="muted">Running 20 passes and building the heatmap. This can take several seconds.</p>}
            {status !== "ready" && !busy && <p className="muted">Analyze turns on when the server reports that the model is ready.</p>}
            {error && <p className="error" role="alert">{error}</p>}
          </section>

          {result && (
            <>
              <section className={`verdict ${result.needs_review ? "review" : "ok"}`} aria-live="polite">
                {result.needs_review ? (
                  <>
                    <h2 className="verdict-title">Low confidence</h2>
                    <p className="verdict-note">The model's repeated passes disagree too much. Do not rely on this result; ask an expert.</p>
                    <p className="muted">Best guess: {result.label} ({pct(result.confidence)})</p>
                  </>
                ) : (
                  <>
                    <h2 className="verdict-title">{result.label}</h2>
                    <p className="verdict-note">Confidence {pct(result.confidence)}. The model's passes agree.</p>
                  </>
                )}
              </section>

              <section className="panel">
                <h2>How sure is it?</h2>
                <UncertaintyGauge value={result.uncertainty} threshold={result.threshold} review={result.needs_review} />
                <ProbabilityBars probabilities={result.probabilities} />
                <div className="actions">
                  <a className="button secondary" href={`data:image/png;base64,${result.heatmap}`} download="grad-cam.png">
                    Download heatmap
                  </a>
                  <button type="button" className="secondary" onClick={copySummary}>
                    {copied ? "Copied" : "Copy summary"}
                  </button>
                </div>
                {result.seconds !== undefined && <p className="muted small">Prediction took {result.seconds} s.</p>}
              </section>

              <details className="panel help">
                <summary>How to read this</summary>
                <dl>
                  <dt>Confidence</dt>
                  <dd>How much probability the model gives its top answer.</dd>
                  <dt>Uncertainty</dt>
                  <dd>The model answers 20 times with part of its network randomly switched off. If the answers differ, uncertainty is high. Above the limit, the app asks for expert review instead of answering.</dd>
                  <dt>Grad-CAM</dt>
                  <dd>Red areas are the regions that pushed the model toward its answer. It shows where the model looked, not proof that it is right.</dd>
                </dl>
              </details>
            </>
          )}
        </aside>
      </main>

      {history.length > 0 && (
        <section className="history" aria-label="Earlier results in this session">
          <h2>This session</h2>
          <ul>
            {history.map((h) => (
              <li key={h.id}>
                <button type="button" onClick={() => openFromHistory(h)} className={image?.url === h.url ? "current" : undefined}>
                  <img src={h.url} alt="" />
                  <span className="h-label">{h.result.needs_review ? "Needs review" : h.result.label}</span>
                  <span className="h-meta">uncertainty {h.result.uncertainty.toFixed(2)}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <footer>Educational project. Not a medical device and not for diagnosis.</footer>
    </div>
  );
}
