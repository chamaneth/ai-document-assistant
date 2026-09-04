const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');

let mainWindow;
let pythonProcess = null;
const BACKEND_PORT = 8000;

// 1. Production / Development Companion Sidecar Launcher
function startPythonSidecar() {
  const isProd = app.isPackaged;
  let execPath;
  let execArgs = [];
  let cwdPath;

  if (isProd) {
    const binaryName = process.platform === 'win32' ? 'rag_backend.exe' : 'rag_backend';
    execPath = path.join(process.resourcesPath, 'rag_backend', binaryName);
    cwdPath = path.join(process.resourcesPath, 'rag_backend');
  } else {
    const backendPath = path.join(__dirname, '..', 'backend');
    execPath = path.join(backendPath, '.venv', 'Scripts', 'python.exe');
    execArgs = ['-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', BACKEND_PORT.toString()];
    cwdPath = backendPath;
  }

  console.log(`[Security Sidecar] Launching Python companion process (${isProd ? 'Production Binary' : 'Development Server'})...`);

  try {
    if (fs.existsSync(execPath)) {
      pythonProcess = spawn(execPath, execArgs, {
        cwd: cwdPath,
        shell: false,
        env: { 
          ...process.env, 
          PYTHONUNBUFFERED: '1',
          API_SECRET_KEY: 'local_sec_token_984712839'
        }
      });

      if (!isProd) {
        pythonProcess.stdout?.on('data', (data) => console.log(`[FastAPI Sidecar]: ${data}`));
        pythonProcess.stderr?.on('data', (data) => console.error(`[FastAPI Sidecar Log]: ${data}`));
      }

      pythonProcess.on('close', (code) => {
        console.log(`[Sidecar] Companion process exited with code ${code}`);
      });
    } else {
      console.warn(`[Sidecar Warning] Executable not found at ${execPath}. Waiting for dev server...`);
    }
  } catch (err) {
    console.error('[Sidecar Error] Failed to launch companion process:', err);
  }
}

// 2. Poll Backend Health
function waitForBackend(callback, retries = 35) {
  if (retries === 0) {
    console.error('[Backend Timeout] FastAPI server did not respond in time.');
    callback(false);
    return;
  }

  http.get(`http://127.0.0.1:${BACKEND_PORT}/health`, (res) => {
    if (res.statusCode === 200) {
      console.log('[Security Sidecar] FastAPI Backend verified healthy!');
      callback(true);
    } else {
      setTimeout(() => waitForBackend(callback, retries - 1), 1000);
    }
  }).on('error', () => {
    setTimeout(() => waitForBackend(callback, retries - 1), 1000);
  });
}

// 3. Create Main Electron Window with Production Path Mapping
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 700,
    frame: true,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#0F0F11',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      allowRunningInsecureContent: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png')
  });

  // Security Hardening: Block external popup window creation
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https:') || url.startsWith('http:')) {
      shell.openExternal(url);
    }
    return { action: 'deny' };
  });

  // Security Hardening: Prevent unhandled internal navigation
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('http://localhost') && !url.startsWith('file://')) {
      event.preventDefault();
    }
  });

  const isProd = app.isPackaged;
  if (!isProd) {
    mainWindow.loadURL('http://localhost:3000');
  } else {
    // Production Assets Path Mapping (process.resourcesPath/frontend_dist/index.html)
    const prodHtmlPath = path.join(process.resourcesPath, 'frontend_dist', 'index.html');
    const fallbackPath = path.join(__dirname, '..', 'frontend', 'dist', 'index.html');

    if (fs.existsSync(prodHtmlPath)) {
      mainWindow.loadFile(prodHtmlPath);
    } else if (fs.existsSync(fallbackPath)) {
      mainWindow.loadFile(fallbackPath);
    } else {
      console.error("[Production Error] Could not locate static frontend index.html");
    }
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// App Lifecycle
app.on('ready', () => {
  startPythonSidecar();
  waitForBackend((success) => {
    createWindow();
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    console.log('[Sidecar] Terminating Python companion process...');
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
