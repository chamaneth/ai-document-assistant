import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Cpu, HardDrive, ShieldCheck, Database, Layers, RefreshCw, Trash2, Activity, Server, FileText, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { API_BASE_URL } from '../config';

export default function AdminDashboard({ onClearDB }) {
  const [telemetry, setTelemetry] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  const fetchTelemetry = async () => {
    setRefreshing(true);
    try {
      const res = await axios.get(`${API_BASE_URL}/admin/stats`);
      setTelemetry(res.data);
    } catch (err) {
      console.error("Error fetching admin telemetry:", err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 10000);
    return () => clearInterval(interval);
  }, []);

  const handlePurgeUploads = async () => {
    try {
      const res = await axios.post(`${API_BASE_URL}/admin/purge_uploads`);
      setStatusMessage(`Successfully purged ${res.data.purged_files_count} temporary file(s).`);
      fetchTelemetry();
    } catch (err) {
      setStatusMessage('Failed to purge temporary files.');
    }
  };

  if (loading && !telemetry) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center h-[calc(100vh-3.5rem)] text-cyber-muted">
        <Activity className="w-8 h-8 text-blue-400 animate-spin mb-3" />
        <p className="text-xs font-mono">Loading System Telemetry & Storage Metrics...</p>
      </div>
    );
  }

  const sys = telemetry?.system || {};
  const storage = telemetry?.storage || {};
  const kb = telemetry?.knowledge_base || {};
  const models = telemetry?.models || {};

  return (
    <main className="flex-1 overflow-y-auto p-6 bg-cyber-bg space-y-6 h-[calc(100vh-3.5rem)]">
      {/* Top Header & Actions */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white font-['Outfit'] flex items-center gap-2">
            <Server className="w-6 h-6 text-blue-400" /> Admin & System Telemetry
          </h2>
          <p className="text-xs text-cyber-muted mt-0.5">
            Monitor RAM memory usage, model cache footprints, vector storage metrics, and security audit logs.
          </p>
        </div>

        <button
          onClick={fetchTelemetry}
          disabled={refreshing}
          className="px-3 py-2 rounded-xl glass-panel text-xs font-medium text-cyber-text hover:border-blue-500/40 flex items-center space-x-2 transition-all"
        >
          <RefreshCw className={`w-4 h-4 text-blue-400 ${refreshing ? 'animate-spin' : ''}`} />
          <span>Refresh Telemetry</span>
        </button>
      </div>

      {statusMessage && (
        <div className="p-3 rounded-xl bg-blue-500/10 border border-blue-500/20 text-xs text-blue-300 flex items-center justify-between">
          <span className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-blue-400" /> {statusMessage}
          </span>
          <button onClick={() => setStatusMessage('')} className="text-cyber-muted hover:text-white text-xs">✕</button>
        </div>
      )}

      {/* Grid Row 1: Resource Telemetry & Storage */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* System Memory & CPU */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-5 rounded-2xl border-cyber-border space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyber-muted flex items-center gap-2">
              <Cpu className="w-4 h-4 text-blue-400" /> System Resources
            </h3>
            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono border border-emerald-500/20">
              {sys.platform} ({sys.cpu_cores} Cores)
            </span>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-cyber-muted">RAM Memory Usage</span>
                <span className="font-mono text-white font-semibold">{sys.ram_used_gb} GB / {sys.ram_total_gb} GB ({sys.ram_usage_percent}%)</span>
              </div>
              <div className="w-full h-2 rounded-full bg-cyber-card overflow-hidden border border-cyber-border">
                <div 
                  className={`h-full rounded-full transition-all duration-500 ${
                    sys.ram_usage_percent > 85 ? 'bg-rose-500' : sys.ram_usage_percent > 65 ? 'bg-amber-400' : 'bg-gradient-to-r from-blue-500 to-cyan-400'
                  }`}
                  style={{ width: `${sys.ram_usage_percent}%` }}
                />
              </div>
            </div>

            <div className="pt-2 border-t border-cyber-border flex justify-between text-[11px] font-mono text-cyber-muted">
              <span>Python Runtime: {sys.python_version}</span>
              <span>OS Release: {sys.platform_release}</span>
            </div>
          </div>
        </motion.div>

        {/* Local Storage Footprint */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-panel p-5 rounded-2xl border-cyber-border space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyber-muted flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-cyan-400" /> Storage Footprint
            </h3>
            <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 font-mono">
              Offline Cache
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border">
              <p className="text-[10px] text-cyber-muted">Models Cache</p>
              <p className="text-xs font-bold font-mono text-cyan-400 mt-1">{storage.models_cache_formatted}</p>
            </div>
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border">
              <p className="text-[10px] text-cyber-muted">Vector Database</p>
              <p className="text-xs font-bold font-mono text-blue-400 mt-1">{storage.chroma_db_formatted}</p>
            </div>
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border">
              <p className="text-[10px] text-cyber-muted">Uploads Folder</p>
              <p className="text-xs font-bold font-mono text-amber-400 mt-1">{storage.uploads_formatted}</p>
            </div>
          </div>
        </motion.div>

        {/* Knowledge Base Overview */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-panel p-5 rounded-2xl border-cyber-border space-y-4"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyber-muted flex items-center gap-2">
              <Layers className="w-4 h-4 text-purple-400" /> Knowledge Base Index
            </h3>
            <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 font-mono">
              ChromaDB Store
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border">
              <p className="text-[10px] text-cyber-muted">Indexed PDFs</p>
              <p className="text-lg font-bold font-mono text-white mt-0.5">{kb.indexed_documents_count}</p>
            </div>
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border">
              <p className="text-[10px] text-cyber-muted">Total Pages</p>
              <p className="text-lg font-bold font-mono text-purple-400 mt-0.5">{kb.total_pages}</p>
            </div>
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border">
              <p className="text-[10px] text-cyber-muted">Total Sections</p>
              <p className="text-lg font-bold font-mono text-cyan-400 mt-0.5">{kb.total_chunks}</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Grid Row 2: AI Pipeline & Security Audit */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Pipeline Specs */}
        <div className="glass-panel p-5 rounded-2xl border-cyber-border space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-cyber-muted flex items-center gap-2">
            <Database className="w-4 h-4 text-emerald-400" /> AI Model Pipeline
          </h3>

          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-xl bg-cyber-card/60 border border-cyber-border flex items-center justify-between">
              <div>
                <p className="text-[10px] text-cyber-muted">Embedding Model</p>
                <p className="font-mono text-white font-medium mt-0.5">{models.embedding_model}</p>
              </div>
              <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-[10px]">Cached (.models/)</span>
            </div>

            <div className="p-3 rounded-xl bg-cyber-card/60 border border-cyber-border flex items-center justify-between">
              <div>
                <p className="text-[10px] text-cyber-muted">Neural LLM Pipeline</p>
                <p className="font-mono text-white font-medium mt-0.5">{models.llm_model}</p>
              </div>
              <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px] font-mono">{models.pipeline_type}</span>
            </div>
          </div>
        </div>

        {/* Security Audit Log */}
        <div className="glass-panel p-5 rounded-2xl border-cyber-border space-y-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-cyber-muted flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-blue-400" /> Security & Privacy Audit
          </h3>

          <div className="space-y-2 text-xs">
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border flex items-center justify-between">
              <span className="text-cyber-muted">Offline Network Privacy</span>
              <span className="text-emerald-400 font-semibold text-[11px]">100% Offline (No Cloud Data Sent)</span>
            </div>
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border flex items-center justify-between">
              <span className="text-cyber-muted">Security Headers (CSP / HSTS)</span>
              <span className="text-blue-400 font-mono text-[11px]">Active</span>
            </div>
            <div className="p-2.5 rounded-xl bg-cyber-card/60 border border-cyber-border flex items-center justify-between">
              <span className="text-cyber-muted">Path Traversal & Sanitization</span>
              <span className="text-blue-400 font-mono text-[11px]">Active</span>
            </div>
          </div>
        </div>
      </div>

      {/* Admin Actions Footer */}
      <div className="p-4 glass-panel rounded-2xl border-cyber-border flex items-center justify-between bg-cyber-card/40">
        <div>
          <h4 className="text-xs font-bold text-white">Cache & Storage Management</h4>
          <p className="text-[11px] text-cyber-muted">Clean temporary files or purge document knowledge base.</p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={handlePurgeUploads}
            className="px-3 py-2 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 text-xs font-semibold flex items-center gap-2 transition-all"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Purge Uploads Folder</span>
          </button>

          <button
            onClick={onClearDB}
            className="px-3 py-2 rounded-xl bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-semibold flex items-center gap-2 transition-all"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Knowledge Base</span>
          </button>
        </div>
      </div>
    </main>
  );
}
