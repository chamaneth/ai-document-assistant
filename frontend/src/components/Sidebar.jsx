import React, { useState } from 'react';
import { UploadCloud, FileText, Trash2, Layers, CheckCircle2, AlertCircle, RefreshCw, PlusCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Sidebar({ onUploadPDF, onClearDB, onDeleteDoc, onOpenPasteModal, indexedDocs, uploading, statusMessage }) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUploadPDF(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onUploadPDF(e.target.files[0]);
    }
  };

  return (
    <aside className="w-80 glass-panel border-r border-cyber-border flex flex-col justify-between p-4 z-10 select-none bg-cyber-bg/90">
      <div className="space-y-5">
        {/* Dropzone Header */}
        <div className="space-y-2">
          <h2 className="text-xs font-bold uppercase tracking-wider text-cyber-muted flex items-center justify-between">
            <span>ADD DOCUMENT</span>
            <span className="text-[10px] text-cyan-400 font-mono">PDF, DOCX, TXT, MD, CSV...</span>
          </h2>

          <form
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onSubmit={(e) => e.preventDefault()}
            className={`relative flex flex-col items-center justify-center p-5 border-2 border-dashed rounded-2xl transition-all cursor-pointer overflow-hidden ${
              dragActive 
                ? 'border-blue-500 bg-blue-500/10 scale-[1.02]' 
                : 'border-cyber-border bg-cyber-card/40 hover:border-blue-500/50 hover:bg-cyber-card/80'
            }`}
          >
            <input
              type="file"
              accept=".pdf,.txt,.md,.docx,.csv,.json,.html,.rtf"
              onChange={handleChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            
            <div className="w-9 h-9 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-2">
              <UploadCloud className="w-5 h-5 text-blue-400" />
            </div>

            <p className="text-xs font-semibold text-white text-center">
              Drag & Drop document here
            </p>
            <p className="text-[11px] text-cyber-muted text-center mt-0.5">
              PDF, DOCX, TXT, MD, CSV, JSON, HTML
            </p>
          </form>

          {/* Quick Paste Raw Text Button */}
          <button
            onClick={onOpenPasteModal}
            className="w-full py-2 px-3 rounded-xl border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 text-xs font-medium flex items-center justify-center space-x-2 transition-all"
          >
            <PlusCircle className="w-4 h-4 text-cyan-400" />
            <span>+ Paste Raw Text / Note</span>
          </button>
        </div>

        {/* Upload Status Notification */}
        <AnimatePresence>
          {(uploading || statusMessage) && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-3 rounded-xl bg-cyber-card border border-cyber-border text-xs flex items-start space-x-2"
            >
              {uploading ? (
                <RefreshCw className="w-4 h-4 text-blue-400 animate-spin flex-shrink-0 mt-0.5" />
              ) : statusMessage.includes('Error') ? (
                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
              )}
              <span className="text-cyber-muted font-mono leading-tight">{statusMessage || 'Reading & preparing document...'}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Saved Documents Section */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-cyber-muted flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              SAVED DOCUMENTS ({indexedDocs.length})
            </h3>
          </div>

          <div className="space-y-2 max-h-[250px] overflow-y-auto pr-1">
            {indexedDocs.length === 0 ? (
              <div className="p-4 rounded-xl border border-cyber-border/40 bg-cyber-card/20 text-center">
                <p className="text-xs text-cyber-muted">No documents uploaded yet.</p>
                <p className="text-[10px] text-cyber-muted/70 mt-1">Upload a PDF, Word doc, or paste a note to start asking questions.</p>
              </div>
            ) : (
              indexedDocs.map((doc, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="p-3 rounded-xl bg-cyber-card/60 border border-cyber-border hover:border-blue-500/40 flex items-center justify-between group transition-all"
                >
                  <div className="flex items-center space-x-3 overflow-hidden">
                    <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-4 h-4 text-blue-400" />
                    </div>
                    <div className="truncate">
                      <p className="text-xs font-semibold text-white truncate">{doc.filename}</p>
                      <p className="text-[10px] text-cyber-muted font-mono">{doc.pages} page/section • {doc.chunks} chunks</p>
                    </div>
                  </div>

                  <button
                    onClick={() => onDeleteDoc(doc.filename)}
                    className="p-1.5 rounded-lg text-cyber-muted hover:text-rose-400 hover:bg-rose-500/10 opacity-60 group-hover:opacity-100 transition-all"
                    title={`Delete ${doc.filename}`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </motion.div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Footer Clear All Button */}
      <div className="pt-4 border-t border-cyber-border">
        <button
          onClick={onClearDB}
          disabled={indexedDocs.length === 0}
          className="w-full py-2.5 px-4 rounded-xl border border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 font-semibold text-xs transition-all flex items-center justify-center space-x-2 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Trash2 className="w-4 h-4" />
          <span>Clear All Documents</span>
        </button>
      </div>
    </aside>
  );
}
