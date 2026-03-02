import { useState, useRef, useEffect, useCallback } from "react";

const MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX";
const MODEL_SIZE = "~650MB";
const STORAGE_KEY = "qwen35-cache-checked";

// Check cache for model files
async function isModelCached() {
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    return keys.some(
      (req) =>
        req.url.includes("Qwen3.5-0.8B-ONNX") ||
        req.url.includes("Qwen3.5-0.8B-ONNX".replace("/", "%2F"))
    );
  } catch {
    return false;
  }
}

function LabHeader() {
  const homeUrl = import.meta.env.DEV
    ? "https://localhost:8030/"
    : "https://lab.kortexa.ai/";

  return (
    <header className="absolute top-0 left-0 right-0 p-6 z-10">
      <a href={homeUrl} className="back-link">
        <img src={`${homeUrl}lab-transparent.png`} alt="" className="logo" />
        <span className="text-sm font-medium uppercase tracking-wider">kortexa.ai lab</span>
        <span className="text-sm">&larr;</span>
      </a>
    </header>
  );
}

function PulseDots() {
  return (
    <span className="inline-flex gap-1 ml-2">
      <span className="pulse-dot" />
      <span className="pulse-dot" />
      <span className="pulse-dot" />
    </span>
  );
}

function ProgressBar({ progress }) {
  if (!progress || !progress.total) return null;
  const pct = Math.round((progress.loaded / progress.total) * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-neutral-500">
        <span>{progress.file || "Loading..."}</span>
        <span>{pct}%</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function parseThinking(content) {
  if (!content.includes("<think>") && "<think>".startsWith(content.trim())) {
    return { thinking: null, reply: "", thinkingDone: false };
  }

  if (!content.includes("<think>")) {
    return { thinking: null, reply: cleanSpecialTokens(content), thinkingDone: true };
  }

  const afterOpen = content.split("<think>")[1];

  if (afterOpen.includes("</think>")) {
    const [thinking, ...rest] = afterOpen.split("</think>");
    return { thinking: cleanSpecialTokens(thinking.trim()), reply: cleanSpecialTokens(rest.join("</think>").trim()), thinkingDone: true };
  }

  return { thinking: cleanSpecialTokens(afterOpen.trim()), reply: "", thinkingDone: false };
}

function cleanSpecialTokens(text) {
  return text
    .replace(/<\|[a-z_]+\|>/g, "")
    .trim();
}

function ImageLightbox({ src, onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="lightbox" onClick={onClose}>
      <img src={src} className="lightbox-img" onClick={(e) => e.stopPropagation()} />
    </div>
  );
}

function ThinkingBlock({ thinking }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="think-block mb-3">
      <button
        onClick={() => setOpen(!open)}
        className="think-toggle"
      >
        <span className="think-icon">{open ? "▾" : "▸"}</span>
        <span>Thought for a moment</span>
      </button>
      {open && (
        <div className="think-content">
          {thinking}
        </div>
      )}
    </div>
  );
}

function AssistantMessage({ content, isStreaming }) {
  const { thinking, reply, thinkingDone } = parseThinking(content);
  const stillThinking = isStreaming && thinking && !thinkingDone;

  if (stillThinking) {
    return (
      <div className="message message-assistant">
        <span className="text-neutral-400 text-xs font-medium">Thinking</span>
        <PulseDots />
      </div>
    );
  }

  return (
    <div className="message message-assistant">
      {thinking && <ThinkingBlock thinking={thinking} />}
      {reply ? (
        <div className="whitespace-pre-wrap">{reply}</div>
      ) : (
        isStreaming && <PulseDots />
      )}
    </div>
  );
}

function checkWebGPU() {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

function isMobile() {
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export default function App() {
  const [status, setStatus] = useState("checking"); // checking | idle | loading | ready | error
  const [statusText, setStatusText] = useState("");
  const [progress, setProgress] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [stats, setStats] = useState(null);
  const [attachedImage, setAttachedImage] = useState(null); // { dataUrl, width, height }
  const [lightboxSrc, setLightboxSrc] = useState(null);

  const workerRef = useRef(null);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const streamRef = useRef("");
  const fileInputRef = useRef(null);

  // Attach image from File object
  const attachImage = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        setAttachedImage({ dataUrl: e.target.result, width: img.width, height: img.height });
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleWorkerMessage = useCallback((e) => {
    const { type, data } = e.data;

    switch (type) {
      case "status":
        setStatusText(data);
        break;
      case "progress":
        if (data.status === "progress") {
          setProgress({ file: data.file, loaded: data.loaded, total: data.total });
        }
        break;
      case "loaded":
        setStatus("ready");
        setStatusText("");
        setProgress(null);
        break;
      case "generate_start":
        streamRef.current = "";
        break;
      case "token": {
        streamRef.current += data.text;
        const text = streamRef.current;
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === "assistant") {
            updated[updated.length - 1] = { ...last, content: text };
          } else {
            updated.push({ role: "assistant", content: text });
          }
          return updated;
        });
        setStats({ tokensPerSec: data.tokensPerSec, tokenCount: data.tokenCount });
        break;
      }
      case "generate_done":
        setGenerating(false);
        setStats({
          tokensPerSec: data.tokensPerSec,
          tokenCount: data.tokenCount,
          elapsed: data.elapsed,
        });
        setTimeout(() => inputRef.current?.focus(), 100);
        break;
      case "error":
        setStatus("error");
        setStatusText(data);
        setGenerating(false);
        break;
    }
  }, []);

  // Create worker and check cache on mount
  useEffect(() => {
    if (isMobile()) {
      setStatus("idle");
      return;
    }

    if (!checkWebGPU()) {
      setStatus("error");
      setStatusText("WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.");
      return;
    }

    let cancelled = false;
    isModelCached().then((cached) => {
      if (cancelled) return;

      const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
      worker.addEventListener("message", handleWorkerMessage);
      workerRef.current = worker;

      if (cached) {
        setStatus("loading");
        setStatusText("Loading from cache...");
        worker.postMessage({ type: "load" });
      } else {
        setStatus("idle");
      }
    });

    return () => {
      cancelled = true;
      if (workerRef.current) {
        workerRef.current.removeEventListener("message", handleWorkerMessage);
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [handleWorkerMessage]);

  const loadModel = useCallback(() => {
    setStatus("loading");
    setStatusText("Downloading model...");
    if (!workerRef.current) {
      const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
      worker.addEventListener("message", handleWorkerMessage);
      workerRef.current = worker;
    }
    workerRef.current.postMessage({ type: "load" });
  }, [handleWorkerMessage]);

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text || generating || status !== "ready") return;

    const userMsg = attachedImage
      ? { role: "user", content: text, image: attachedImage }
      : { role: "user", content: text };

    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setAttachedImage(null);
    setGenerating(true);
    setStats(null);

    workerRef.current?.postMessage({
      type: "generate",
      data: {
        messages: newMessages
          .filter((m) => ["user", "assistant"].includes(m.role))
          .map(({ role, content, image }) => ({
            role,
            content,
            ...(image ? { image: image.dataUrl } : {}),
          })),
        maxTokens: 2048,
      },
    });
  }, [input, generating, status, messages, attachedImage]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Paste image from clipboard
  const handlePaste = useCallback((e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        attachImage(item.getAsFile());
        return;
      }
    }
  }, [attachImage]);

  const hasWebGPU = checkWebGPU();
  const mobile = isMobile();

  return (
    <div className="page">
      <LabHeader />

      <section className="hero pt-24 pb-4">
        <p className="eyebrow">In-Browser VLM</p>
        <h1 className="title">Qwen 3.5 0.8B WebGPU</h1>
        <p className="lede">
          Run Qwen 3.5 vision-language model entirely in your browser. Text chat + image understanding — no server, no API keys.
        </p>
      </section>

      <section className="content">
        {/* Mobile block */}
        {mobile && (
          <div className="panel text-center space-y-3">
            <p className="text-lg font-semibold">Desktop Required</p>
            <p className="text-sm text-neutral-500">
              Loading a 0.8B parameter VLM needs more memory than mobile browsers can handle.
              Try this on a desktop with Chrome or Edge.
            </p>
          </div>
        )}

        {!mobile && (
          <div className="space-y-4">
            {/* Checking cache */}
            {status === "checking" && (
              <div className="panel text-center">
                <span className="text-sm text-neutral-500">Checking for cached model</span>
                <PulseDots />
              </div>
            )}

            {/* Download prompt */}
            {status === "idle" && (
              <div className="panel text-center space-y-4">
                <p className="text-sm text-neutral-500">
                  {hasWebGPU ? (
                    <>Ready to download <strong>Qwen 3.5 0.8B VLM</strong> ({MODEL_SIZE}).</>
                  ) : (
                    <span className="text-red-600">
                      WebGPU not available. Use Chrome 113+ or Edge 113+.
                    </span>
                  )}
                </p>
                <button
                  onClick={loadModel}
                  disabled={!hasWebGPU}
                  className="px-6 py-3 bg-neutral-800 text-white rounded-xl font-medium hover:bg-neutral-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Download & Load Model
                </button>
                <p className="text-xs text-neutral-400">
                  Model weights are cached in your browser after first download.
                </p>
              </div>
            )}

            {/* Loading */}
            {status === "loading" && (
              <div className="panel space-y-4">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{statusText}</span>
                  <PulseDots />
                </div>
                <ProgressBar progress={progress} />
              </div>
            )}

            {/* Error */}
            {status === "error" && (
              <div className="panel border-red-200 bg-red-50/80 space-y-3">
                <p className="text-sm text-red-700 font-medium">Something went wrong</p>
                <p className="text-sm text-red-600 font-mono break-all">{statusText}</p>
                <button
                  onClick={() => { setStatus("idle"); setStatusText(""); }}
                  className="text-sm text-red-600 underline hover:text-red-800"
                >
                  Try again
                </button>
              </div>
            )}

            {/* Chat Interface */}
            {status === "ready" && (
              <div className="space-y-4">
                {/* Stats */}
                {stats && (
                  <div className="flex justify-center gap-2">
                    <span className="stats-badge">
                      {stats.tokensPerSec?.toFixed(1)} tok/s
                    </span>
                    {stats.tokenCount && (
                      <span className="stats-badge">
                        {stats.tokenCount} tokens
                      </span>
                    )}
                    {stats.elapsed && (
                      <span className="stats-badge">
                        {stats.elapsed.toFixed(1)}s
                      </span>
                    )}
                  </div>
                )}

                {/* Messages */}
                <div
                  className="chat-container min-h-[200px] max-h-[60vh] overflow-y-auto panel"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => { e.preventDefault(); attachImage(e.dataTransfer.files[0]); }}
                >
                  {messages.length === 0 && (
                    <p className="text-sm text-neutral-400 text-center py-8">
                      Model loaded. Type a message or attach an image to start chatting.
                    </p>
                  )}
                  {messages.map((msg, i) =>
                    msg.role === "user" ? (
                      <div key={i} className="message message-user">
                        {msg.image && (
                          <img
                            src={msg.image.dataUrl}
                            alt="Attached"
                            className="chat-image-thumb"
                            onClick={() => setLightboxSrc(msg.image.dataUrl)}
                          />
                        )}
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      </div>
                    ) : (
                      <AssistantMessage
                        key={i}
                        content={msg.content}
                        isStreaming={generating && i === messages.length - 1}
                      />
                    )
                  )}
                  {generating && messages[messages.length - 1]?.role !== "assistant" && (
                    <div className="message message-assistant">
                      <PulseDots />
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Image preview */}
                {attachedImage && (
                  <div className="flex items-start gap-3 px-4 py-3 bg-neutral-50 border border-neutral-200 rounded-xl">
                    <img
                      src={attachedImage.dataUrl}
                      alt="Preview"
                      className="w-16 h-16 rounded-lg object-cover"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-neutral-500">
                        {attachedImage.width} x {attachedImage.height}
                      </div>
                    </div>
                    <button
                      onClick={() => setAttachedImage(null)}
                      className="text-neutral-400 hover:text-neutral-600 text-lg leading-none"
                      title="Remove image"
                    >
                      x
                    </button>
                  </div>
                )}

                {/* Input */}
                <div className="flex gap-3">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => { attachImage(e.target.files[0]); e.target.value = ""; }}
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={generating}
                    className="px-3 py-3.5 bg-white border border-neutral-200 rounded-xl hover:bg-neutral-50 transition-colors disabled:opacity-40 text-neutral-500"
                    title="Attach image (or paste from clipboard)"
                  >
                    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="2" y="2" width="14" height="14" rx="2" />
                      <circle cx="6" cy="7" r="1.5" />
                      <path d="M2 13l4-4 3 3 2-2 5 5" />
                    </svg>
                  </button>
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onPaste={handlePaste}
                    placeholder="Type a message... (paste or attach image for VLM)"
                    disabled={generating}
                    rows={1}
                    className="flex-1 px-5 py-3.5 bg-white border border-neutral-200 rounded-xl resize-none focus:outline-none focus:border-neutral-400 transition-colors disabled:opacity-50 text-sm"
                  />
                  <button
                    onClick={sendMessage}
                    disabled={generating || !input.trim()}
                    className="px-5 py-3.5 bg-neutral-800 text-white rounded-xl font-medium hover:bg-neutral-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed text-sm"
                  >
                    {generating ? "..." : "Send"}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {lightboxSrc && <ImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />}
    </div>
  );
}
