import React, { useState } from 'react';
import { X, FileText, Check, Plus } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function PasteTextModal({ isOpen, onClose, onAddText }) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [submitting, setSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!content.trim() || submitting) return;
    setSubmitting(true);
    onAddText(title.trim() || 'Pasted_Note', content.trim());
    setSubmitting(false);
    setTitle('');
    setContent('');
    onClose();
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="w-full max-w-xl glass-panel rounded-2xl border-cyber-border shadow-2xl overflow-hidden bg-cyber-bg"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-cyber-border bg-cyber-card/60">
            <div className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-cyan-400" />
              <h3 className="font-bold text-sm text-white font-['Outfit']">Paste Raw Text / Note</h3>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-white/10 text-cyber-muted hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Form Content */}
          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            <div>
              <label className="block text-xs font-semibold text-cyber-muted mb-1">
                Note Title / Name
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Project Brief, Legal Clause, Meeting Notes..."
                className="w-full py-2.5 px-4 rounded-xl bg-cyber-card/80 border border-cyber-border text-white text-xs placeholder-cyber-muted focus:outline-none focus:border-cyan-400"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-cyber-muted mb-1">
                Paste Text Content *
              </label>
              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                rows={8}
                required
                placeholder="Paste raw text, email snippets, clauses, or notes here..."
                className="w-full p-4 rounded-xl bg-cyber-card/80 border border-cyber-border text-white text-xs placeholder-cyber-muted focus:outline-none focus:border-cyan-400 font-mono resize-none"
              />
            </div>

            <div className="pt-2 flex items-center justify-between border-t border-cyber-border">
              <span className="text-[11px] text-cyber-muted font-mono">
                Stored safely in local document library
              </span>

              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-3 py-1.5 rounded-xl text-xs text-cyber-muted hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!content.trim() || submitting}
                  className="px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-cyber-glow flex items-center gap-1.5 transition-all disabled:opacity-40"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add to Document Library</span>
                </button>
              </div>
            </div>
          </form>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
