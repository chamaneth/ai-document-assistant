import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChatWorkspace from './components/ChatWorkspace';
import AdminDashboard from './components/AdminDashboard';
import SettingsModal from './components/SettingsModal';
import PasteTextModal from './components/PasteTextModal';
import LicenseModal from './components/LicenseModal';
import LoadingScreen from './components/LoadingScreen';
import { API_BASE_URL, API_SECRET_KEY } from './config';

axios.defaults.headers.common['X-API-Key'] = API_SECRET_KEY;

export default function App() {
  const [initialLoading, setInitialLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('chat');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isPasteModalOpen, setIsPasteModalOpen] = useState(false);
  const [isLicenseModalOpen, setIsLicenseModalOpen] = useState(false);
  const [showAdminPanel, setShowAdminPanel] = useState(false);
  const [licenseInfo, setLicenseInfo] = useState(null);
  
  const [appSettings, setAppSettings] = useState({
    theme: 'cyber-dark',
    topK: 3,
    maxLength: 512,
    autoScroll: true,
    showCitationsDefault: true
  });

  const [backendConnected, setBackendConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [indexedDocs, setIndexedDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [loadingQuery, setLoadingQuery] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    const saved = localStorage.getItem('ai_doc_assistant_settings');
    if (saved) {
      try {
        setAppSettings(JSON.parse(saved));
      } catch (e) {}
    }
  }, []);

  const handleSaveSettings = (newSettings) => {
    setAppSettings(newSettings);
    localStorage.setItem('ai_doc_assistant_settings', JSON.stringify(newSettings));
  };

  const checkBackendHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/health`);
      if (res.data.status === 'healthy') {
        setBackendConnected(true);
        setShowAdminPanel(!!res.data.enable_admin_panel);
        fetchIndexedDocs();
        fetchLicenseStatus();
      }
    } catch (err) {
      setBackendConnected(false);
    }
  };

  const fetchLicenseStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/license/status`);
      setLicenseInfo(res.data);
    } catch (err) {
      console.error("Error fetching license status:", err);
    }
  };

  const fetchIndexedDocs = async () => {
    try {
      const res = await axios.get(`${API_BASE_URL}/indexed_docs`);
      setIndexedDocs(res.data.documents || []);
    } catch (err) {
      console.error("Error fetching indexed docs:", err);
    }
  };

  useEffect(() => {
    checkBackendHealth();
    fetchLicenseStatus();
    const interval = setInterval(checkBackendHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleUploadPDF = async (file) => {
    if (!file) return;
    setUploading(true);
    setStatusMessage(`Uploading & processing '${file.name}'...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_BASE_URL}/upload_file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (res.data.status === 'success') {
        setStatusMessage(`Successfully saved '${file.name}'!`);
        fetchIndexedDocs();
        
        setMessages(prev => [
          ...prev,
          {
            sender: 'bot',
            text: `📄 Document **${file.name}** (${res.data.document.pages} page/section) added to your document library. You can now ask questions!`,
            citations: []
          }
        ]);
      }
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.message || 'Failed to upload document.';
      setStatusMessage(`Error: ${errMsg}`);
    } finally {
      setUploading(false);
    }
  };

  // Raw Text Note Upload Handler
  const handleUploadRawText = async (title, content) => {
    setUploading(true);
    setStatusMessage(`Indexing note '${title}'...`);

    try {
      const res = await axios.post(`${API_BASE_URL}/upload_text`, { title, content });
      if (res.data.status === 'success') {
        setStatusMessage(`Successfully added note '${res.data.document.filename}'!`);
        fetchIndexedDocs();

        setMessages(prev => [
          ...prev,
          {
            sender: 'bot',
            text: `📝 Quick Note **${res.data.document.filename}** indexed in your document library. You can now ask questions about it!`,
            citations: []
          }
        ]);
      }
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.message || 'Failed to add raw text note.';
      setStatusMessage(`Error: ${errMsg}`);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDoc = async (filename) => {
    try {
      setStatusMessage(`Deleting '${filename}'...`);
      await axios.delete(`${API_BASE_URL}/document/${encodeURIComponent(filename)}`);
      fetchIndexedDocs();
      setStatusMessage(`Deleted '${filename}' from library.`);
    } catch (err) {
      setStatusMessage(`Failed to delete '${filename}'.`);
    }
  };

  const handleSendMessage = async (questionText) => {
    if (!questionText.trim()) return;

    const userMsg = { sender: 'user', text: questionText };
    setMessages(prev => [...prev, userMsg]);
    setLoadingQuery(true);

    try {
      const res = await axios.post(`${API_BASE_URL}/query`, {
        question: questionText,
        chat_history: messages.map(m => ({ sender: m.sender, text: m.text })),
        top_k: appSettings.topK || 3,
        max_length: appSettings.maxLength || 512
      });

      const botMsg = {
        sender: 'bot',
        text: res.data.answer,
        citations: res.data.citations || []
      };

      setMessages(prev => [...prev, botMsg]);
      fetchLicenseStatus();
    } catch (err) {
      const errMsg = err.response?.data?.detail || err.message || 'Failed to get answer.';
      setMessages(prev => [
        ...prev,
        {
          sender: 'bot',
          text: `⚠️ ${errMsg}`,
          citations: []
        }
      ]);
      if (typeof errMsg === 'string' && errMsg.includes('TRIAL_LIMIT')) {
        fetchLicenseStatus();
        setIsLicenseModalOpen(true);
      }
    } finally {
      setLoadingQuery(false);
    }
  };

  const handleClearDB = async () => {
    try {
      await axios.post(`${API_BASE_URL}/clear_db`);
      setIndexedDocs([]);
      setMessages([]);
      setStatusMessage('All documents cleared from library.');
    } catch (err) {
      setStatusMessage('Failed to clear documents.');
    }
  };

  const handleExportChat = () => {
    if (messages.length === 0) return;

    let mdContent = `# AI Document Assistant — Research Report\n\n`;
    mdContent += `**Date**: ${new Date().toLocaleString()}\n`;
    mdContent += `**Indexed Documents**: ${indexedDocs.map(d => d.filename).join(', ') || 'None'}\n\n`;
    mdContent += `---\n\n`;

    messages.forEach((msg) => {
      if (msg.sender === 'user') {
        mdContent += `### ❓ Question\n> ${msg.text}\n\n`;
      } else {
        mdContent += `### 🤖 Answer\n${msg.text}\n\n`;
        if (msg.citations && msg.citations.length > 0) {
          mdContent += `#### 📌 Page References:\n`;
          msg.citations.forEach((c) => {
            mdContent += `- **${c.source}** (Page ${c.page}): "${c.content}"\n`;
          });
          mdContent += `\n`;
        }
      }
    });

    const blob = new Blob([mdContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI_Research_Report_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (initialLoading) {
    return <LoadingScreen onComplete={() => setInitialLoading(false)} />;
  }

  const themeBgMap = {
    'cyber-dark': 'bg-[#0F0F11]',
    'midnight-navy': 'bg-[#0B132B]',
    'neon-violet': 'bg-[#120B2E]'
  };
  const activeBgClass = themeBgMap[appSettings.theme] || 'bg-[#0F0F11]';

  return (
    <div className={`h-screen w-screen flex flex-col ${activeBgClass} overflow-hidden font-sans transition-colors duration-300`}>
      <Header 
        backendConnected={backendConnected} 
        showAdminPanel={showAdminPanel}
        activeTab={activeTab} 
        setActiveTab={setActiveTab}
        onOpenSettings={() => setIsSettingsOpen(true)}
        licenseInfo={licenseInfo}
        onOpenLicense={() => setIsLicenseModalOpen(true)}
      />
      
      {activeTab === 'chat' || !showAdminPanel ? (
        <div className="flex flex-1 overflow-hidden">
          <Sidebar
            onUploadPDF={handleUploadPDF}
            onClearDB={handleClearDB}
            onDeleteDoc={handleDeleteDoc}
            onOpenPasteModal={() => setIsPasteModalOpen(true)}
            indexedDocs={indexedDocs}
            uploading={uploading}
            statusMessage={statusMessage}
          />

          <ChatWorkspace
            messages={messages}
            onSendMessage={handleSendMessage}
            onExportChat={handleExportChat}
            loading={loadingQuery}
            hasDocs={indexedDocs.length > 0}
            settings={appSettings}
            licenseInfo={licenseInfo}
            onOpenLicense={() => setIsLicenseModalOpen(true)}
          />
        </div>
      ) : (
        <AdminDashboard onClearDB={handleClearDB} />
      )}

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={appSettings}
        onSaveSettings={handleSaveSettings}
      />

      <PasteTextModal
        isOpen={isPasteModalOpen}
        onClose={() => setIsPasteModalOpen(false)}
        onAddText={handleUploadRawText}
      />

      <LicenseModal
        isOpen={isLicenseModalOpen}
        onClose={() => setIsLicenseModalOpen(false)}
        licenseInfo={licenseInfo}
        onLicenseUpdated={fetchLicenseStatus}
      />
    </div>
  );
}
