import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Key, ShieldCheck, CheckCircle2, AlertCircle, X, ExternalLink, Sparkles } from 'lucide-react';
import axios from 'axios';
import { API_BASE_URL } from '../config';

export default function LicenseModal({ isOpen, onClose, licenseInfo, onLicenseUpdated }) {
  const [licenseKey, setLicenseKey] = useState('');
  const [registeredTo, setRegisteredTo] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  if (!isOpen) return null;

  const handleActivate = async (e) => {
    e.preventDefault();
    if (!licenseKey.trim()) return;

    setLoading(true);
    setErrorMsg('');
    setSuccessMsg('');

    try {
      const res = await axios.post(`${API_BASE_URL}/license/activate`, {
        license_key: licenseKey.trim(),
        registered_to: registeredTo.trim() || 'Verified Customer'
      });

      if (res.data.valid) {
        setSuccessMsg(res.data.message || 'Software successfully activated!');
        setTimeout(() => {
          onLicenseUpdated && onLicenseUpdated();
          onClose();
        }, 1200);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Invalid or expired license key.';
      setErrorMsg(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleDeactivate = async () => {
    if (!confirm('Are you sure you want to deactivate this license on this device?')) return;
    setLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/license/deactivate`);
      onLicenseUpdated && onLicenseUpdated();
      setSuccessMsg('License removed. Switched to evaluation mode.');
    } catch (err) {
      setErrorMsg('Failed to deactivate license.');
    } finally {
      setLoading(false);
    }
  };

  const isLicensed = licenseInfo?.is_licensed;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="w-full max-w-lg glass-panel rounded-3xl border-cyber-border shadow-2xl overflow-hidden bg-cyber-bg"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-5 border-b border-cyber-border bg-cyber-card/60">
            <div className="flex items-center space-x-2.5">
              <div className="w-8 h-8 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
                <Key className="w-4 h-4 text-cyan-400" />
              </div>
              <div>
                <h3 className="font-bold text-sm text-white font-['Outfit']">
                  Software License & Registration
                </h3>
                <p className="text-[11px] text-cyber-muted">
                  Commercial offline license validation
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-xl hover:bg-white/10 text-cyber-muted hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 space-y-6">
            {/* Status Card */}
            <div className={`p-4 rounded-2xl border flex items-start space-x-3.5 ${
              isLicensed 
                ? 'bg-emerald-500/10 border-emerald-500/30' 
                : 'bg-blue-500/10 border-blue-500/30'
            }`}>
              {isLicensed ? (
                <ShieldCheck className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
              ) : (
                <Sparkles className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
              )}
              <div className="space-y-1 flex-1">
                <div className="flex items-center justify-between">
                  <span className={`text-xs font-bold font-mono uppercase tracking-wider ${
                    isLicensed ? 'text-emerald-400' : 'text-cyan-300'
                  }`}>
                    {licenseInfo?.tier_name || 'Evaluation Trial'}
                  </span>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-medium ${
                    isLicensed ? 'bg-emerald-500/20 text-emerald-300' : 'bg-blue-500/20 text-blue-300'
                  }`}>
                    {isLicensed ? 'Active Lifetime' : 'Evaluation'}
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {isLicensed 
                    ? `Registered to ${licenseInfo?.registered_to || 'Verified Owner'} (${licenseInfo?.license_key})`
                    : 'Running in trial mode. Enter your commercial license key to unlock unrestricted document indexing and export capabilities.'}
                </p>
              </div>
            </div>

            {/* Notifications */}
            {successMsg && (
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs text-emerald-300 flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                <span>{successMsg}</span>
              </div>
            )}

            {errorMsg && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-xs text-rose-300 flex items-center space-x-2">
                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Activation Form */}
            <form onSubmit={handleActivate} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-cyber-muted mb-1.5">
                  License Key *
                </label>
                <input
                  type="text"
                  value={licenseKey}
                  onChange={(e) => setLicenseKey(e.target.value)}
                  placeholder="e.g. AIDA-STD-9F3B2A1C-7E4A19"
                  required
                  className="w-full py-3 px-4 rounded-xl bg-cyber-card/80 border border-cyber-border text-white text-xs font-mono placeholder-cyber-muted focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 uppercase transition-all"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-cyber-muted mb-1.5">
                  Registered Name / Organization (Optional)
                </label>
                <input
                  type="text"
                  value={registeredTo}
                  onChange={(e) => setRegisteredTo(e.target.value)}
                  placeholder="e.g. Acorn Legal LLP or Jane Doe"
                  className="w-full py-2.5 px-4 rounded-xl bg-cyber-card/80 border border-cyber-border text-white text-xs placeholder-cyber-muted focus:outline-none focus:border-cyan-400 transition-all"
                />
              </div>

              <div className="pt-2 flex items-center justify-between">
                <a
                  href="https://gumroad.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[11px] text-cyan-400 hover:text-cyan-300 font-medium flex items-center gap-1 transition-colors"
                >
                  <span>Don't have a key? Purchase License</span>
                  <ExternalLink className="w-3 h-3" />
                </a>

                <div className="flex items-center space-x-2">
                  {isLicensed && (
                    <button
                      type="button"
                      onClick={handleDeactivate}
                      disabled={loading}
                      className="px-3 py-2 rounded-xl text-xs text-rose-400 hover:bg-rose-500/10 border border-rose-500/30 transition-all"
                    >
                      Deactivate
                    </button>
                  )}
                  <button
                    type="submit"
                    disabled={loading || !licenseKey.trim()}
                    className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-cyber-glow flex items-center gap-1.5 transition-all disabled:opacity-40"
                  >
                    {loading ? 'Activating...' : 'Activate License'}
                  </button>
                </div>
              </div>
            </form>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
