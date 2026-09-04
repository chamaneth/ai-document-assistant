import os
import sys
import platform
import psutil
from typing import Dict, Any
from fastapi import APIRouter, Depends
from core.security import verify_api_key
from core.config import settings
from core.services.rag_service import rag_service

router = APIRouter(tags=["Admin & Telemetry"])

def get_dir_size_bytes(dir_path: str) -> int:
    total_size = 0
    if os.path.exists(dir_path):
        for dirpath, dirnames, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
    return total_size

@router.get("/admin/stats", dependencies=[Depends(verify_api_key)])
def get_admin_telemetry() -> Dict[str, Any]:
    # System RAM Telemetry
    memory = psutil.virtual_memory()
    
    # Calculate storage footprints
    models_size = get_dir_size_bytes(settings.MODELS_DIR)
    chroma_size = get_dir_size_bytes(settings.CHROMA_DIR)
    uploads_size = get_dir_size_bytes(settings.UPLOADS_DIR)

    # Total chunks count across indexed docs
    total_chunks = sum(doc.get("chunks", 0) for doc in rag_service.indexed_documents)
    total_pages = sum(doc.get("pages", 0) for doc in rag_service.indexed_documents)

    return {
        "system": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python_version": sys.version.split()[0],
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(memory.total / (1024 ** 3), 2),
            "ram_used_gb": round(memory.used / (1024 ** 3), 2),
            "ram_usage_percent": memory.percent
        },
        "storage": {
            "models_cache_bytes": models_size,
            "models_cache_formatted": f"{round(models_size / (1024 ** 2), 1)} MB",
            "chroma_db_bytes": chroma_size,
            "chroma_db_formatted": f"{round(chroma_size / (1024 ** 2), 1)} MB",
            "uploads_bytes": uploads_size,
            "uploads_formatted": f"{round(uploads_size / (1024 ** 2), 1)} MB"
        },
        "knowledge_base": {
            "indexed_documents_count": len(rag_service.indexed_documents),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "documents": rag_service.indexed_documents
        },
        "models": {
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "llm_model": settings.LLM_MODEL_NAME,
            "offline_privacy": "100% Local (No External API Transmissions)",
            "pipeline_type": "Seq2SeqLM (Text2Text Generation)"
        },
        "security": {
            "api_key_auth": "Enforced",
            "security_headers": "Active",
            "path_sanitization": "Active"
        }
    }

@router.post("/admin/purge_uploads", dependencies=[Depends(verify_api_key)])
def purge_uploads() -> Dict[str, Any]:
    purged_count = 0
    if os.path.exists(settings.UPLOADS_DIR):
        for fname in os.listdir(settings.UPLOADS_DIR):
            fpath = os.path.join(settings.UPLOADS_DIR, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                purged_count += 1
    return {"status": "success", "purged_files_count": purged_count}
