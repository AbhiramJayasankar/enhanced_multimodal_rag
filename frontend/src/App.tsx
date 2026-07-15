import React, { useState, useEffect, useRef } from "react";
import { 
  FileText, 
  Send, 
  Database, 
  Layers, 
  UploadCloud, 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Sparkles, 
  Search,
  BookOpen,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Clock
} from "lucide-react";

// API Base URL
const API_BASE = "http://127.0.0.1:8000";

interface SystemStatus {
  markdown_files: string[];
  image_files: string[];
  chunks_count: number;
  indexed_texts: number;
  indexed_images: number;
  api_configured: boolean;
}

interface ProcessStatus {
  status: "idle" | "running" | "completed" | "error";
  message: string;
  step: string;
  processed: number;
  total: number;
  error_msg: string;
}

interface TextResult {
  text: string;
  distance: number;
}

interface ImageResult {
  filename: string;
  distance: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  retrieval_info?: {
    enhanced_queries: string[];
    texts: TextResult[];
    images: ImageResult[];
  };
}

export default function App() {
  // Application State
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [processStatus, setProcessStatus] = useState<ProcessStatus>({
    status: "idle",
    message: "",
    step: "",
    processed: 0,
    total: 0,
    error_msg: ""
  });
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  
  const [chunkingLoading, setChunkingLoading] = useState(false);
  const [activeSources, setActiveSources] = useState<{[key: number]: "text" | "image" | "query"}>({});
  const [expandedSources, setExpandedSources] = useState<{[key: number]: boolean}>({});

  // Files upload state
  const [markdownFile, setMarkdownFile] = useState<File | null>(null);
  const [imageFiles, setImageFiles] = useState<FileList | null>(null);
  const [uploadProgress, setUploadProgress] = useState("");

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load metrics initially
  useEffect(() => {
    fetchStatus();
    checkProcessStatus();
  }, []);

  // Poll embedding builder status when it is running
  useEffect(() => {
    let timer: number;
    if (processStatus.status === "running") {
      timer = window.setInterval(() => {
        checkProcessStatus();
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [processStatus.status]);

  // Scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, chatLoading]);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      console.error("Failed to fetch system status:", e);
    }
  };

  const checkProcessStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/process/status`);
      const data = await res.json();
      setProcessStatus(data);
      
      // If completed successfully, update metrics too
      if (data.status === "completed") {
        fetchStatus();
      }
    } catch (e) {
      console.error("Failed to check process status:", e);
    }
  };

  // Upload handlers
  const handleMarkdownUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!markdownFile) return;

    setUploadProgress("Uploading markdown...");
    const formData = new FormData();
    formData.append("file", markdownFile);

    try {
      const res = await fetch(`${API_BASE}/api/upload/markdown`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        setUploadProgress("Markdown uploaded successfully!");
        setMarkdownFile(null);
        fetchStatus();
      } else {
        const err = await res.json();
        setUploadProgress(`Error: ${err.detail}`);
      }
    } catch (err) {
      setUploadProgress("Upload failed.");
    }
  };

  const handleImagesUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageFiles || imageFiles.length === 0) return;

    setUploadProgress(`Uploading ${imageFiles.length} images...`);
    let uploadedCount = 0;

    for (let i = 0; i < imageFiles.length; i++) {
      const file = imageFiles[i];
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${API_BASE}/api/upload/image`, {
          method: "POST",
          body: formData,
        });
        if (res.ok) {
          uploadedCount++;
          setUploadProgress(`Uploaded ${uploadedCount}/${imageFiles.length} images...`);
        }
      } catch (err) {
        console.error("Image upload failed for file:", file.name, err);
      }
    }

    setUploadProgress(`Uploaded ${uploadedCount} images successfully!`);
    setImageFiles(null);
    fetchStatus();
  };

  // Run Chunking
  const triggerChunking = async () => {
    setChunkingLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/process/chunk`, { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        alert(data.message);
        fetchStatus();
      } else {
        alert(`Error: ${data.detail}`);
      }
    } catch (e) {
      alert("Failed to start chunking.");
    } finally {
      setChunkingLoading(false);
    }
  };

  // Run Embedding Building
  const triggerEmbedding = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/process/embed`, { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        checkProcessStatus();
      } else {
        alert(`Error: ${data.detail}`);
      }
    } catch (e) {
      alert("Failed to start embedding building.");
    }
  };

  // Chat query
  const sendChatMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userQuery = chatInput;
    setChatInput("");
    setChatLoading(true);

    // Add user query to chat flow
    const newMsg: Message = { role: "user", content: userQuery };
    setMessages(prev => [...prev, newMsg]);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userQuery }),
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: data.answer,
          retrieval_info: {
            enhanced_queries: data.enhanced_queries,
            texts: data.texts,
            images: data.images
          }
        }]);
      } else {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: `⚠️ Error executing pipeline: ${data.detail}`
        }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "⚠️ Failed to communicate with the backend server."
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Accordion Toggles
  const toggleSourceExpand = (idx: number) => {
    setExpandedSources(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
    if (!activeSources[idx]) {
      setActiveSources(prev => ({ ...prev, [idx]: "text" }));
    }
  };

  const setSourceTab = (idx: number, tab: "text" | "image" | "query") => {
    setActiveSources(prev => ({
      ...prev,
      [idx]: tab
    }));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Top Navigation */}
      <header style={{
        display: "flex", 
        justifyContent: "space-between", 
        alignItems: "center", 
        padding: "1rem 2rem", 
        backgroundColor: "var(--bg-dark)", 
        borderBottom: "1px solid var(--border-subtle)"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{
            backgroundColor: "var(--primary)", 
            padding: "0.5rem", 
            borderRadius: "8px", 
            boxShadow: "0 0 10px var(--primary-glow)"
          }}>
            <Sparkles size={20} color="#fff" />
          </div>
          <span style={{ fontSize: "1.25rem", fontWeight: "700", letterSpacing: "-0.5px" }}>
            Gemini Multimodal RAG Studio
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <button 
            onClick={fetchStatus} 
            className="btn-primary" 
            style={{ padding: "0.5rem 0.75rem", fontSize: "0.875rem", backgroundColor: "var(--bg-panel-light)" }}
          >
            <RefreshCw size={14} /> Refresh metrics
          </button>
          <div style={{ fontSize: "0.875rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "0.25rem" }}>
            <Clock size={14} /> Server Status: 
            <span style={{ color: "var(--success)", fontWeight: "600" }}>● Online</span>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left Side: Sidebar */}
        <div className="sidebar">
          {/* Section: Metrics Panel */}
          <div className="glass-panel" style={{ padding: "1.25rem" }}>
            <h3 style={{ margin: "0 0 1rem 0", fontSize: "1rem", fontWeight: "700", display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-main)" }}>
              <Database size={16} color="var(--primary)" /> Database State
            </h3>
            
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
              <div style={{ backgroundColor: "var(--bg-deep)", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Markdown files</div>
                <div style={{ fontSize: "1.5rem", fontWeight: "700", marginTop: "0.25rem" }}>
                  {status ? status.markdown_files.length : "..."}
                </div>
              </div>
              <div style={{ backgroundColor: "var(--bg-deep)", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Corpus Images</div>
                <div style={{ fontSize: "1.5rem", fontWeight: "700", marginTop: "0.25rem" }}>
                  {status ? status.image_files.length : "..."}
                </div>
              </div>
              <div style={{ backgroundColor: "var(--bg-deep)", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Parsed Chunks</div>
                <div style={{ fontSize: "1.5rem", fontWeight: "700", marginTop: "0.25rem" }}>
                  {status ? status.chunks_count : "..."}
                </div>
              </div>
              <div style={{ backgroundColor: "var(--bg-deep)", padding: "0.75rem", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Vector Indexes</div>
                <div style={{ fontSize: "1rem", fontWeight: "700", marginTop: "0.25rem", color: "var(--primary)" }}>
                  {status ? `${status.indexed_texts} T / ${status.indexed_images} I` : "..."}
                </div>
              </div>
            </div>
          </div>

          {/* Section: File Uploader */}
          <div className="glass-panel" style={{ padding: "1.25rem" }}>
            <h3 style={{ margin: "0 0 1rem 0", fontSize: "1rem", fontWeight: "700", display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-main)" }}>
              <UploadCloud size={16} color="var(--primary)" /> Document Ingestion
            </h3>
            
            {/* Upload status message */}
            {uploadProgress && (
              <div style={{ 
                backgroundColor: "var(--bg-deep)", 
                border: "1px solid var(--border-subtle)", 
                padding: "0.5rem", 
                borderRadius: "6px", 
                fontSize: "0.8125rem", 
                marginBottom: "1rem",
                color: "var(--primary)"
              }}>
                {uploadProgress}
              </div>
            )}

            {/* Markdown Uploader */}
            <form onSubmit={handleMarkdownUpload} style={{ marginBottom: "1rem" }}>
              <label style={{ display: "block", fontSize: "0.8125rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Add Markdown File (.md)
              </label>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <input 
                  type="file" 
                  accept=".md" 
                  onChange={(e) => setMarkdownFile(e.target.files?.[0] || null)}
                  style={{ fontSize: "0.8125rem", flexGrow: 1 }}
                />
                <button type="submit" disabled={!markdownFile} className="btn-primary" style={{ padding: "0.25rem 0.75rem", fontSize: "0.8125rem" }}>
                  Upload
                </button>
              </div>
            </form>

            {/* Image Uploader */}
            <form onSubmit={handleImagesUpload}>
              <label style={{ display: "block", fontSize: "0.8125rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                Add Corpus Images (.png, .jpg)
              </label>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <input 
                  type="file" 
                  accept=".png,.jpg,.jpeg" 
                  multiple 
                  onChange={(e) => setImageFiles(e.target.files)}
                  style={{ fontSize: "0.8125rem", flexGrow: 1 }}
                />
                <button type="submit" disabled={!imageFiles || imageFiles.length === 0} className="btn-primary" style={{ padding: "0.25rem 0.75rem", fontSize: "0.8125rem" }}>
                  Upload
                </button>
              </div>
            </form>
          </div>

          {/* Section: Task Processor */}
          <div className="glass-panel" style={{ padding: "1.25rem" }}>
            <h3 style={{ margin: "0 0 1rem 0", fontSize: "1rem", fontWeight: "700", display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-main)" }}>
              <Layers size={16} color="var(--primary)" /> Process Manager
            </h3>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              <button 
                onClick={triggerChunking} 
                disabled={chunkingLoading || processStatus.status === "running"} 
                className="btn-primary" 
                style={{ justifyContent: "center", width: "100%", backgroundColor: "var(--bg-panel-light)" }}
              >
                <FileText size={16} /> {chunkingLoading ? "Chunking..." : "1. Chunk Markdown Files"}
              </button>

              <button 
                onClick={triggerEmbedding} 
                disabled={processStatus.status === "running"} 
                className="btn-primary" 
                style={{ justifyContent: "center", width: "100%" }}
              >
                <Play size={16} /> {processStatus.status === "running" ? "Building..." : "2. Build Vector Embeddings"}
              </button>
            </div>

            {/* Embedding progress panel */}
            {processStatus.status !== "idle" && (
              <div style={{ 
                marginTop: "1.25rem", 
                padding: "0.75rem", 
                backgroundColor: "var(--bg-deep)", 
                border: "1px solid var(--border-subtle)", 
                borderRadius: "8px" 
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.875rem", fontWeight: "600" }}>
                  {processStatus.status === "running" && <RefreshCw size={14} className="pulsing-border" style={{ animation: "spin 2s linear infinite" }} />}
                  {processStatus.status === "completed" && <CheckCircle size={14} color="var(--success)" />}
                  {processStatus.status === "error" && <AlertCircle size={14} color="var(--error)" />}
                  {processStatus.message}
                </div>

                {processStatus.status === "running" && (
                  <div style={{ marginTop: "0.5rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      <span>Step: {processStatus.step === "text" ? "Texts Embedding" : "Images Embedding"}</span>
                      <span>{processStatus.processed} / {processStatus.total}</span>
                    </div>
                    {/* Progress Bar container */}
                    <div style={{ width: "100%", backgroundColor: "var(--bg-panel)", height: "6px", borderRadius: "3px", marginTop: "0.25rem", overflow: "hidden" }}>
                      <div style={{ 
                        width: `${processStatus.total > 0 ? (processStatus.processed / processStatus.total) * 100 : 0}%`, 
                        backgroundColor: "var(--primary)", 
                        height: "100%", 
                        transition: "width 0.3s ease" 
                      }} />
                    </div>
                  </div>
                )}

                {processStatus.status === "error" && (
                  <div style={{ fontSize: "0.75rem", color: "var(--error)", marginTop: "0.25rem" }}>
                    {processStatus.error_msg}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Chat Container */}
        <div className="main-content">
          <div className="chat-container">
            {/* Chat Title / Header */}
            <div style={{ 
              padding: "1rem 1.5rem", 
              backgroundColor: "var(--bg-deep)", 
              borderBottom: "1px solid var(--border-subtle)",
              display: "flex",
              alignItems: "center",
              gap: "0.5rem"
            }}>
              <BookOpen size={18} color="var(--primary)" />
              <span style={{ fontWeight: "600" }}>Multimodal Document Context Chat</span>
            </div>

            {/* Chat Messages */}
            <div className="chat-history">
              {messages.length === 0 ? (
                <div style={{ 
                  display: "flex", 
                  flexDirection: "column", 
                  alignItems: "center", 
                  justifyContent: "center", 
                  height: "100%", 
                  color: "var(--text-muted)",
                  textAlign: "center",
                  gap: "1rem",
                  padding: "2rem"
                }}>
                  <Sparkles size={48} color="var(--primary)" style={{ opacity: 0.6 }} />
                  <div>
                    <h4 style={{ margin: 0, color: "var(--text-main)", fontSize: "1.125rem", fontWeight: "600" }}>No Messages Yet</h4>
                    <p style={{ fontSize: "0.875rem", marginTop: "0.25rem" }}>
                      Load markdown and images, chunk, index, and start chatting with your knowledge base!
                    </p>
                  </div>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div key={idx} style={{ display: "flex", flexDirection: "column", width: "100%" }}>
                    <div className={msg.role === "user" ? "chat-bubble-user" : "chat-bubble-assistant"}>
                      {msg.content}
                    </div>

                    {/* Show expandable source files if assistant and retrieval info is present */}
                    {msg.role === "assistant" && msg.retrieval_info && (
                      <div style={{ 
                        alignSelf: "flex-start", 
                        width: "85%", 
                        marginTop: "0.25rem", 
                        backgroundColor: "var(--bg-deep)",
                        border: "1px solid var(--border-subtle)",
                        borderRadius: "8px",
                        overflow: "hidden"
                      }}>
                        {/* Header accordion toggle */}
                        <div 
                          onClick={() => toggleSourceExpand(idx)}
                          style={{ 
                            padding: "0.5rem 1rem", 
                            display: "flex", 
                            justifyContent: "space-between", 
                            alignItems: "center", 
                            fontSize: "0.8125rem", 
                            color: "var(--text-muted)",
                            cursor: "pointer",
                            userSelect: "none"
                          }}
                        >
                          <span style={{ display: "flex", alignItems: "center", gap: "0.25rem", fontWeight: "600" }}>
                            <Search size={12} color="var(--primary)" /> 
                            Show Retrieved Sources & Vectors
                          </span>
                          {expandedSources[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        </div>

                        {/* Accordion expanded content */}
                        {expandedSources[idx] && (
                          <div style={{ borderTop: "1px solid var(--border-subtle)", padding: "1rem" }}>
                            {/* Tabs selectors */}
                            <div style={{ display: "flex", gap: "0.5rem", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "0.5rem", marginBottom: "0.75rem" }}>
                              <button 
                                onClick={() => setSourceTab(idx, "text")}
                                className={`source-tab-btn ${activeSources[idx] === "text" ? "active" : ""}`}
                              >
                                📝 Passages
                              </button>
                              <button 
                                onClick={() => setSourceTab(idx, "image")}
                                className={`source-tab-btn ${activeSources[idx] === "image" ? "active" : ""}`}
                              >
                                🖼️ Visual Pages
                              </button>
                              <button 
                                onClick={() => setSourceTab(idx, "query")}
                                className={`source-tab-btn ${activeSources[idx] === "query" ? "active" : ""}`}
                              >
                                🔮 Expanded Queries
                              </button>
                            </div>

                            {/* Tab Content */}
                            {activeSources[idx] === "text" && (
                              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                                {msg.retrieval_info.texts.map((t, t_idx) => (
                                  <div key={t_idx} style={{ backgroundColor: "var(--bg-panel)", padding: "0.75rem", borderRadius: "6px", borderLeft: "3px solid var(--primary)" }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.25rem" }}>
                                      <span>Passage {t_idx + 1}</span>
                                      <span>Distance L2: {t.distance.toFixed(4)}</span>
                                    </div>
                                    <div style={{ fontSize: "0.8125rem", whiteSpace: "pre-wrap", color: "var(--text-main)" }}>
                                      {t.text}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {activeSources[idx] === "image" && (
                              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: "0.75rem" }}>
                                {msg.retrieval_info.images.map((img, i_idx) => (
                                  <div key={i_idx} style={{ backgroundColor: "var(--bg-panel)", padding: "0.5rem", borderRadius: "6px", textAlign: "center" }}>
                                    <img 
                                      src={`${API_BASE}/images/${img.filename}`} 
                                      alt="source match" 
                                      style={{ width: "100%", height: "auto", borderRadius: "4px", marginBottom: "0.5rem" }} 
                                      onError={(e) => {
                                        // Fallback if image load fails
                                        (e.target as HTMLElement).style.display = "none";
                                      }}
                                    />
                                    <div style={{ fontSize: "0.75rem", fontWeight: "600" }}>
                                      {img.filename}
                                    </div>
                                    <div style={{ fontSize: "0.6875rem", color: "var(--text-muted)" }}>
                                      Dist: {img.distance.toFixed(4)}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {activeSources[idx] === "query" && (
                              <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                                {msg.retrieval_info.enhanced_queries.map((q, q_idx) => (
                                  <div key={q_idx} style={{ fontSize: "0.8125rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                                    <span style={{ color: "var(--primary)" }}>✦</span> {q}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              )}

              {/* Chat loading state placeholder */}
              {chatLoading && (
                <div style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-muted)", fontSize: "0.875rem" }}>
                  <RefreshCw size={14} className="pulsing-border" style={{ animation: "spin 2s linear infinite" }} />
                  Gemini is thinking and searching...
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input form */}
            <form onSubmit={sendChatMessage} style={{ 
              padding: "1rem 1.5rem", 
              backgroundColor: "var(--bg-deep)", 
              borderTop: "1px solid var(--border-subtle)",
              display: "flex",
              gap: "0.75rem"
            }}>
              <input 
                type="text" 
                className="form-input" 
                placeholder="Ask any question about your documents..." 
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                disabled={chatLoading}
              />
              <button type="submit" disabled={chatLoading || !chatInput.trim()} className="btn-primary">
                <Send size={16} /> Send
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
