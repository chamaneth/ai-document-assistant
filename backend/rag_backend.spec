# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all data files and submodules for ChromaDB, LangChain, Transformers, Core, Docx, PyPDF
datas = (
    collect_data_files('chromadb') + 
    collect_data_files('langchain_community') + 
    collect_data_files('transformers') + 
    collect_data_files('sentence_transformers') +
    collect_data_files('docx') +
    [('.env', '.')]
)

hiddenimports = (
    collect_submodules('core') +
    collect_submodules('chromadb') +
    collect_submodules('langchain_community') +
    collect_submodules('langchain_huggingface') +
    collect_submodules('transformers') +
    collect_submodules('sentence_transformers') +
    collect_submodules('docx') +
    collect_submodules('pypdf') +
    ['uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on', 'pydantic', 'fastapi', 'python_multipart', 'dotenv', 'docx', 'pypdf', 'lxml']
)

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='rag_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='rag_backend',
)
