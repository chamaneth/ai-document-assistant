import React, { useState } from 'react';
import { X, Sliders, Palette, Cpu, ShieldCheck, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function SettingsModal({ isOpen, onClose, settings, onSaveSettings }) {
  const [theme, setTheme] = useState(settings?.theme || 'cyber-dark');
  const [topK, setTopK] = useState(settings?.topK || 3);
  const [maxLength, setMaxLength] = useState(settings?.maxLength || 512);
  const [autoScroll, setAutoScroll] = useState(settings?.autoScroll ?? true);
  const [showCitationsDefault, setShowCitationsDefault] = useState(settings?.showCitationsDefault ?? true);
  const [savedSuccess, setSavedSuccess] = useState(false);

  if (!isOpen) return null;

  const handleSave = () => {
    onSaveSettings({
      theme,
      topK,
      maxLength,
      autoScroll,
      showCitationsDefault
    });
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 800);
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="w-full max-w-lg glass-panel rounded-2xl border-cyber-border shadow-2xl overflow-hidden bg-cyber-bg"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-cyber-border bg-cyber-card/60">
            <div className="flex items-center space-x-2">
              <Sliders className="w-5 h-5 text-blue-400" />
              <h3 className="font-bold text-sm text-white font-['Outfit']">Application Preferences & Customization</h3>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-white/10 text-cyber-muted hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Body Options */}
          <div className="p-6 space-y-6 max-h-[75vh] overflow-y-auto">
            {/* 1. Theme Color Accent */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-cyber-muted uppercase tracking-wider flex items-center gap-2">
                <Palette className="w-4 h-4 text-cyan-400" /> Interface Theme Accent
              </label>
              <div className="grid grid-cols-3 gap-3">
                <button
                  type="button"
                  onClick={() => setTheme('cyber-dark')}
                  className={`p-3 rounded-xl border text-xs font-medium flex flex-col items-center gap-2 transition-all ${
                    theme === 'cyber-dark'
                      ? 'border-blue-500 bg-blue-500/10 text-white shadow-cyber-glow'
                      : 'border-cyber-border bg-cyber-card/40 text-cyber-muted hover:text-white'
                  }`}
                >
                  <div className="w-6 h-6 rounded-full bg-[#0F0F11] border border-blue-500 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-blue-500" />
                  </div>
                  <span>Cyber Dark</span>
                </button>

                <button
                  type="button"
                  onClick={() => setTheme('midnight-navy')}
                  className={`p-3 rounded-xl border text-xs font-medium flex flex-col items-center gap-2 transition-all ${
                    theme === 'midnight-navy'
                      ? 'border-cyan-400 bg-cyan-500/10 text-white shadow-cyber-glow'
                      : 'border-cyber-border bg-cyber-card/40 text-cyber-muted hover:text-white'
                  }`}
                >
                  <div className="w-6 h-6 rounded-full bg-[#0B132B] border border-cyan-400 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-cyan-400" />
                  </div>
                  <span>Midnight Navy</span>
                </button>

                <button
                  type="button"
                  onClick={() => setTheme('neon-violet')}
                  className={`p-3 rounded-xl border text-xs font-medium flex flex-col items-center gap-2 transition-all ${
                    theme === 'neon-violet'
                      ? 'border-purple-400 bg-purple-500/10 text-white shadow-cyber-glow'
                      : 'border-cyber-border bg-cyber-card/40 text-cyber-muted hover:text-white'
                  }`}
                >
                  <div className="w-6 h-6 rounded-full bg-[#120B2E] border border-purple-400 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-purple-400" />
                  </div>
                  <span>Neon Violet</span>
                </button>
              </div>
            </div>

            {/* 2. RAG Retrieval Parameters */}
            <div className="space-y-3 pt-2 border-t border-cyber-border">
              <label className="text-xs font-bold text-cyber-muted uppercase tracking-wider flex items-center gap-2">
                <Cpu className="w-4 h-4 text-purple-400" /> AI Retrieval Parameters
              </label>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-white">Retrieved Context Sections (Top-K)</span>
                    <span className="font-mono text-cyan-400 font-bold">{topK} sections</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="w-full h-1.5 rounded-lg bg-cyber-card accent-blue-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-cyber-muted mt-1">
                    Higher Top-K retrieves more document context but takes slightly longer to generate responses.
                  </p>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-white">Max Answer Tokens</span>
                    <span className="font-mono text-purple-400 font-bold">{maxLength} tokens</span>
                  </div>
                  <input
                    type="range"
                    min="256"
                    max="1024"
                    step="128"
                    value={maxLength}
                    onChange={(e) => setMaxLength(parseInt(e.target.value))}
                    className="w-full h-1.5 rounded-lg bg-cyber-card accent-purple-500 cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* 3. Interface Toggles */}
            <div className="space-y-3 pt-2 border-t border-cyber-border">
              <label className="text-xs font-bold text-cyber-muted uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" /> Workspace Options
              </label>

              <div className="space-y-2">
                <label className="flex items-center justify-between p-3 rounded-xl bg-cyber-card/40 border border-cyber-border cursor-pointer hover:border-cyber-border/80">
                  <span className="text-xs text-white">Auto-scroll chat on response</span>
                  <input
                    type="checkbox"
                    checked={autoScroll}
                    onChange={(e) => setAutoScroll(e.target.checked)}
                    className="w-4 h-4 rounded accent-blue-500"
                  />
                </label>

                <label className="flex items-center justify-between p-3 rounded-xl bg-cyber-card/40 border border-cyber-border cursor-pointer hover:border-cyber-border/80">
                  <span className="text-xs text-white">Expand Page References by default</span>
                  <input
                    type="checkbox"
                    checked={showCitationsDefault}
                    onChange={(e) => setShowCitationsDefault(e.target.checked)}
                    className="w-4 h-4 rounded accent-blue-500"
                  />
                </label>
              </div>
            </div>
          </div>

          {/* Footer Save Controls */}
          <div className="p-4 border-t border-cyber-border bg-cyber-card/60 flex items-center justify-between">
            <span className="text-[10px] text-cyber-muted font-mono">Preferences saved locally</span>
            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={onClose}
                className="px-3 py-1.5 rounded-xl text-xs text-cyber-muted hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSave}
                className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-cyber-glow flex items-center gap-1.5 transition-all"
              >
                {savedSuccess ? (
                  <>
                    <Check className="w-4 h-4 text-emerald-300" /> Saved!
                  </>
                ) : (
                  'Save Preferences'
                )}
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
