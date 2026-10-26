import { useEffect, useRef, useState } from "react";
import CompareViewer from "./components/CompareViewer.jsx";
import ProbabilityBars from "./components/ProbabilityBars.jsx";
import UncertaintyGauge from "./components/UncertaintyGauge.jsx";
import "./App.css";

const API =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

const percentage = (value) =>
  `${(Number(value) * 100).toFixed(1)}%`;

const readableLabel = (value) =>
  String(value || "Awaiting analysis").replaceAll("_", " ");

function Icon({ name, ...props }) {
  const paths = {
    dashboard: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    ),
    history: (
      <>
        <path d="M3 11a9 9 0 1 1 2.7 7" />
        <path d="M3 4v7h7M12 7v5l3 2" />
      </>
    ),
    guide: (
      <>
        <path d="M4 3h6a3 3 0 0 1 3 3v15a4 4 0 0 0-4-2H4z" />
        <path d="M20 3h-4a3 3 0 0 0-3 3v15a4 4 0 0 1 4-2h3z" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V3m-5 5 5-5 5 5" />
        <path d="M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" />
      </>
    ),
    scan: (
      <>
        <path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" />
        <circle cx="12" cy="12" r="5" />
        <path d="M12 7v10" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m-5-5 5 5 5-5" />
        <path d="M4 16v5h16v-5" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="12" height="13" rx="2" />
        <path d="M16 8V3H3v13h5" />
      </>
    ),
  };

  return (
    <svg
      width="21"
      height="21"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name] || paths.scan}
    </svg>
  );
}

export default function App() {
  const [section, setSection] = useState("dashboard");
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [view, setView] = useState("original");
  const [status, setStatus] = useState("checking");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [copied, setCopied] = useState(false);

  const fileInput = useRef(null);
  const imageUrls = useRef([]);
  const copyTimer = useRef(null);

  useEffect(() => {
    let stopped = false;
    let timeout;

    async function checkServer() {
      let next = "offline";
      try {
        const response = await fetch(`${API}/health`);
        if (response.ok) {
          const data = await response.json();
          next = data.model_loaded ? "ready" : "no-model";
        }
      } catch {
        next = "offline";
      }
      if (stopped) return;
      setStatus(next);
      timeout = setTimeout(checkServer, next === "ready" ? 30000 : 5000);
    }

    checkServer();
    return () => {
      stopped = true;
      clearTimeout(timeout);
    };
  }, []);

  useEffect(() => {
    return () => {
      imageUrls.current.forEach((url) => URL.revokeObjectURL(url));
      clearTimeout(copyTimer.current);
    };
  }, []);

  function chooseFile(file) {
    if (!file || busy) return;
    if (!file.type.startsWith("image/")) {
      setError("Please select an MRI image.");
      return;
    }
    const url = URL.createObjectURL(file);
    imageUrls.current.push(url);
    setImage({ file, url, name: file.name });
    setResult(null);
    setView("original");
    setError("");
    setCopied(false);
    setSection("dashboard");
  }

  async function analyze() {
    if (!image?.file || busy || status !== "ready") return;
    setBusy(true);
    setError("");
    setResult(null);
    setView("original");
    const selectedImage = image;

    try {
      const body = new FormData();
      body.append("image", selectedImage.file);
      const response = await fetch(`${API}/predict`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Analysis failed.");

      setResult(data);
      setView(data.heatmap ? "split" : "original");
      setHistory((previous) =>
        [
          {
            id: `${Date.now()}-${selectedImage.name}`,
            name: selectedImage.name,
            url: selectedImage.url,
            result: data,
            time: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
          },
          ...previous,
        ].slice(0, 8)
      );
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot reach the server. Please start your backend."
          : err.message || "Something went wrong."
      );
    } finally {
      setBusy(false);
    }
  }

  function openHistory(entry) {
    if (busy) return;
    setImage({ file: null, name: entry.name, url: entry.url });
    setResult(entry.result);
    setView(entry.result.heatmap ? "split" : "original");
    setError("");
    setCopied(false);
    setSection("dashboard");
  }

  async function copySummary() {
    if (!result) return;
    const summary = [
      "Brain MRI analysis — educational project",
      `Image: ${image?.name || "Unknown"}`,
      `Model prediction: ${readableLabel(result.label)}`,
      `Confidence: ${percentage(result.confidence)}`,
      `Uncertainty: ${Number(result.uncertainty).toFixed(2)}`,
      `Review threshold: ${Number(result.threshold).toFixed(2)}`,
      `Expert review required: ${result.needs_review ? "Yes" : "No"}`,
      "Not for medical diagnosis.",
    ].join("\n");

    try {
      await navigator.clipboard.writeText(summary);
      setCopied(true);
      clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Your browser could not copy the summary.");
    }
  }

  const heatmap = result?.heatmap
    ? result.heatmap.startsWith("data:")
      ? result.heatmap
      : `data:image/png;base64,${result.heatmap}`
    : null;

  const statusLabel = {
    checking: "Connecting",
    ready: "Model ready",
    "no-model": "Model not loaded",
    offline: "Server offline",
  }[status];

  const headings = {
    dashboard: ["Dashboard", "Brain Tumor Detection"],
    history: ["Scan history", "Results from this session"],
    guide: ["Result guide", "Understanding model results"],
  };

  return (
    <div className="app-shell">
      <input
        ref={fileInput}
        className="file-input"
        type="file"
        accept="image/*"
        disabled={busy}
        onChange={(event) => {
          chooseFile(event.target.files?.[0]);
          event.target.value = "";
        }}
      />

      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            setSection("dashboard");
          }}
        >
          <span className="brand-icon">
            <Icon name="scan" />
          </span>
          <span>
            <strong>Brain Tumor</strong>
            <small>Detection</small>
          </span>
        </a>

        <nav className="navigation" aria-label="Main sections">
          {[
            ["dashboard", "Dashboard"],
            ["history", "Scan history"],
            ["guide", "Result guide"],
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={`nav-item ${section === key ? "active" : ""}`}
              aria-current={section === key ? "page" : undefined}
              onClick={() => setSection(key)}
            >
              <Icon name={key} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <span className={`status-dot ${status}`} />
          <span>{statusLabel}</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="page-header">
          <div className="header-title">
            <p className="eyebrow">{headings[section][1]}</p>
            <h1>{headings[section][0]}</h1>
          </div>
          <button
            className="button button-primary"
            type="button"
            disabled={busy}
            onClick={() => fileInput.current?.click()}
          >
            <Icon name="upload" />
            <span>Upload MRI</span>
          </button>
        </header>

        {error && (
          <div className="error-message" role="alert">
            {error}
          </div>
        )}

        {section === "dashboard" && (
          <div className="dashboard">
            <section className={`scan-panel ${busy ? "is-analyzing" : ""}`}>
              <div className="scan-heading">
                <div className="scan-heading-text">
                  <h2>Brain mapping</h2>
                  <p title={image?.name}>{image?.name || "No scan selected"}</p>
                </div>
                <span className="scan-badge">MRI</span>
              </div>

              <div className="scan-body">
                {busy && (
                  <div className="scan-animation" aria-hidden="true">
                    <span className="scan-line" />
                    <span className="scan-corner top-left" />
                    <span className="scan-corner top-right" />
                    <span className="scan-corner bottom-left" />
                    <span className="scan-corner bottom-right" />
                  </div>
                )}

                {image ? (
                  <CompareViewer original={image.url} heatmap={heatmap} view={view} />
                ) : (
                  <button
                    type="button"
                    className={`upload-area ${dragging ? "dragging" : ""}`}
                    onClick={() => fileInput.current?.click()}
                    onDragOver={(event) => {
                      event.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(event) => {
                      event.preventDefault();
                      setDragging(false);
                      chooseFile(event.dataTransfer.files?.[0]);
                    }}
                  >
                    <span className="upload-symbol">
                      <Icon name="scan" width="48" height="48" />
                    </span>
                    <strong>Select a brain MRI</strong>
                    <span>Drop an image or browse files</span>
                  </button>
                )}
              </div>

              <div className="scan-toolbar">
                <div className="view-switch" role="group" aria-label="Image display">
                  {[
                    ["original", "Original"],
                    ["split", "Compare"],
                    ["heatmap", "Grad-CAM"],
                  ].map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      aria-pressed={view === key}
                      disabled={!image || (key !== "original" && !heatmap)}
                      onClick={() => setView(key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  className="button button-analyze"
                  disabled={!image?.file || busy || status !== "ready"}
                  onClick={analyze}
                >
                  <Icon name="scan" />
                  {busy ? "Analyzing…" : "Analyze"}
                </button>
              </div>
            </section>

            <section className="insights-panel" aria-live="polite" aria-busy={busy}>
              <div className="section-heading">
                <h2>AI insights</h2>
                <span className="subtle-label">
                  {busy ? "Processing" : result ? "Complete" : "Awaiting scan"}
                </span>
              </div>

              <div className="prediction">
                <p className="eyebrow">Model prediction</p>
                <h3>{readableLabel(result?.label)}</h3>
                <span className={`result-status ${result?.needs_review ? "warning" : ""}`}>
                  {!result
                    ? "No result yet"
                    : result.needs_review
                    ? "Expert review required"
                    : "Model result available"}
                </span>
              </div>

              <div className="probability-section">
                <h3>Class probabilities</h3>
                {result ? (
                  <ProbabilityBars probabilities={result.probabilities} />
                ) : (
                  <div className="empty-probabilities">
                    {["Glioma", "Meningioma", "Pituitary", "No tumor"].map((label) => (
                      <div key={label}>
                        <span>{label}</span>
                        <span className="empty-bar" />
                        <span>—</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="insight-actions">
                <button
                  type="button"
                  className="button button-secondary"
                  disabled={!result}
                  onClick={copySummary}
                >
                  <Icon name="copy" />
                  {copied ? "Copied" : "Copy result"}
                </button>
                {heatmap && (
                  <a
                    className="icon-button"
                    href={heatmap}
                    download="grad-cam.png"
                    title="Download Grad-CAM"
                    aria-label="Download Grad-CAM"
                  >
                    <Icon name="download" />
                  </a>
                )}
              </div>
            </section>

            <div className="metric-grid">
              <section className="metric-card">
                <h2>Confidence</h2>
                <strong className="metric-value">
                  {result ? percentage(result.confidence) : "—"}
                </strong>
                <progress
                  max="1"
                  value={result ? Number(result.confidence) : 0}
                  aria-label="Model confidence"
                />
              </section>

              <section className="metric-card uncertainty-card">
                <h2>Uncertainty</h2>
                {result ? (
                  <UncertaintyGauge
                    value={Number(result.uncertainty)}
                    threshold={Number(result.threshold)}
                    review={Boolean(result.needs_review)}
                  />
                ) : (
                  <>
                    <strong className="metric-value">—</strong>
                    <p>Awaiting analysis</p>
                  </>
                )}
              </section>

              <section className="metric-card">
                <h2>Review status</h2>
                <strong className={`review-value ${result?.needs_review ? "warning-text" : ""}`}>
                  {!result ? "Pending" : result.needs_review ? "Required" : "Not flagged"}
                </strong>
                <p>Clinical interpretation needs a specialist.</p>
              </section>
            </div>
          </div>
        )}

        {section === "history" && (
          <section className="secondary-page">
            <div className="section-heading">
              <h2>Previous scans</h2>
              <span className="count-chip">{history.length} scans</span>
            </div>

            {history.length === 0 ? (
              <div className="empty-state">
                <Icon name="history" width="48" height="48" />
                <h3>No analyzed scans yet</h3>
                <p>Upload an MRI and press Analyze. Results will appear here.</p>
                <button
                  type="button"
                  className="button button-primary"
                  onClick={() => fileInput.current?.click()}
                >
                  <Icon name="upload" />
                  <span>Upload MRI</span>
                </button>
              </div>
            ) : (
              <div className="history-list">
                {history.map((entry) => (
                  <button
                    className="history-row"
                    type="button"
                    key={entry.id}
                    disabled={busy}
                    onClick={() => openHistory(entry)}
                  >
                    <img src={entry.url} alt="" />
                    <span className="history-description">
                      <strong>{entry.name}</strong>
                      <span>
                        {readableLabel(entry.result.label)} ·{" "}
                        {percentage(entry.result.confidence)}
                      </span>
                    </span>
                    <span className={`history-tag ${entry.result.needs_review ? "warning" : ""}`}>
                      {entry.result.needs_review ? "Review" : "OK"}
                    </span>
                    <span className="history-time">{entry.time}</span>
                  </button>
                ))}
              </div>
            )}
          </section>
        )}

        {section === "guide" && (
          <section className="secondary-page guide-page">
            <div className="section-heading">
              <h2>Reading the results</h2>
              <span className="count-chip">4 terms</span>
            </div>

            <dl className="guide-grid">
              <div className="guide-card">
                <dt>Confidence</dt>
                <dd>
                  The probability assigned to the model’s top prediction.
                  High confidence does not guarantee a correct result.
                </dd>
              </div>
              <div className="guide-card">
                <dt>Uncertainty</dt>
                <dd>
                  Measures disagreement between repeated model predictions.
                  Results above the review threshold are flagged for expert review.
                </dd>
              </div>
              <div className="guide-card">
                <dt>Grad-CAM</dt>
                <dd>
                  Highlights regions that influenced the model. It is not a
                  tumor boundary or proof of a tumor.
                </dd>
              </div>
              <div className="guide-card">
                <dt>Review status</dt>
                <dd>
                  “Not flagged” only means the model did not trigger its
                  uncertainty warning. It does not mean the scan is medically normal.
                </dd>
              </div>
            </dl>
          </section>
        )}

        <footer className="page-footer">
          Educational project · Not a medical device or diagnosis.
        </footer>
      </main>
    </div>
  );
}
