import React from 'react';
import { Cpu, MessageSquare, LayoutDashboard, Minus, Square, X, Wifi, WifiOff, Settings, Key } from 'lucide-react';

export default function Header({ 
  backendConnected, 
  showAdminPanel = true, 
  activeTab, 
  setActiveTab, 
  onOpenSettings,
  licenseInfo,
  onOpenLicense
}) {
  const isLicensed = licenseInfo?.is_licensed;

  return (
    <header className="h-14 glass-panel border-b border-cyber-border flex items-center justify-between px-4 select-none z-20 relative">
      {/* Title & Branding */}
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-cyan-400 flex items-center justify-center shadow-cyber-glow">
          <Cpu className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="font-bold text-sm tracking-wide bg-gradient-to-r from-white via-slate-200 to-cyber-muted bg-clip-text text-transparent font-['Outfit']">
              AI DOCUMENT ASSISTANT
            </h1>
            <span className="text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full border bg-blue-500/10 text-blue-400 border-blue-500/20">
              OFFLINE LOCAL
            </span>
          </div>
          <p className="text-[10px] text-cyber-muted font-mono">100% Private Offline AI</p>
        </div>
      </div>

      {/* Navigation View Selector */}
      <div className="flex items-center space-x-1 bg-cyber-card/80 p-1 rounded-xl border border-cyber-border text-xs">
        <button
          onClick={() => setActiveTab('chat')}
          className={`px-3 py-1.5 rounded-lg font-medium flex items-center space-x-2 transition-all ${
            activeTab === 'chat'
              ? 'bg-blue-600 text-white shadow-cyber-glow'
              : 'text-cyber-muted hover:text-white'
          }`}
        >
          <MessageSquare className="w-3.5 h-3.5" />
          <span>Chat Workspace</span>
        </button>

        <button
          onClick={() => setActiveTab('admin')}
          className={`px-3 py-1.5 rounded-lg font-medium flex items-center space-x-2 transition-all ${
            activeTab === 'admin'
              ? 'bg-blue-600 text-white shadow-cyber-glow'
              : 'text-cyber-muted hover:text-white'
          }`}
        >
          <LayoutDashboard className="w-3.5 h-3.5" />
          <span>Admin Dashboard</span>
        </button>

        <button
          onClick={onOpenLicense}
          className={`px-2.5 py-1.5 rounded-lg font-medium text-xs flex items-center space-x-1.5 transition-all border ${
            isLicensed
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20'
              : licenseInfo?.is_trial_locked
                ? 'bg-rose-500/10 text-rose-400 border-rose-500/30 hover:bg-rose-500/20 animate-pulse'
                : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/20'
          }`}
          title="Manage License & Registration"
        >
          <Key className="w-3.5 h-3.5" />
          <span>
            {isLicensed 
              ? (licenseInfo?.tier || 'LICENSED') 
              : licenseInfo?.is_trial_locked
                ? 'Trial Expired — Unlock'
                : `Trial: ${licenseInfo?.trial_queries_remaining ?? 3} Qs Left`}
          </span>
        </button>

        <button
          onClick={onOpenSettings}
          className="p-1.5 rounded-lg text-cyber-muted hover:text-white hover:bg-white/10 transition-all"
          title="Customize & Settings"
        >
          <Settings className="w-4 h-4 text-cyan-400" />
        </button>
      </div>

      {/* Backend Status & Window Controls */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-cyber-card border border-cyber-border text-xs">
          {backendConnected ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-emerald-400 font-mono text-[11px] flex items-center gap-1">
                <Wifi className="w-3 h-3" /> Ready
              </span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span className="text-rose-400 font-mono text-[11px] flex items-center gap-1">
                <WifiOff className="w-3 h-3" /> Offline
              </span>
            </>
          )}
        </div>

        <div className="flex items-center space-x-1 pl-2 border-l border-cyber-border text-cyber-muted">
          <button 
            className="p-1.5 hover:bg-white/10 rounded transition-colors text-cyber-muted hover:text-white"
            title="Minimize Window"
          >
            <Minus className="w-4 h-4" />
          </button>
          <button 
            className="p-1.5 hover:bg-white/10 rounded transition-colors text-cyber-muted hover:text-white"
            title="Maximize Window"
          >
            <Square className="w-3.5 h-3.5" />
          </button>
          <button 
            className="p-1.5 hover:bg-rose-600/80 hover:text-white rounded transition-colors text-cyber-muted"
            title="Close Application"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
}
