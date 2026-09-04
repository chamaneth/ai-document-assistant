from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from core.security import verify_api_key
from core.schemas import RawTextUploadRequest
from core.services.rag_service import rag_service
from core.licensing import can_upload_doc

router = APIRouter(tags=["Document Management"])

@router.post("/upload_pdf", dependencies=[Depends(verify_api_key)])
@router.post("/upload_file", dependencies=[Depends(verify_api_key)])
async def upload_file(file: UploadFile = File(...)):
    current_docs = rag_service.indexed_documents
    if not can_upload_doc(len(current_docs)):
        msg = "TRIAL_LIMIT_EXCEEDED: Your trial has expired. Activate a Lifetime License ($29) to index more documents." if len(current_docs) == 0 else "TRIAL_LIMIT_DOCS: Free evaluation trial allows indexing 1 document. Activate a Lifetime License ($29) to index unlimited documents."
        raise HTTPException(status_code=403, detail=msg)
    return await rag_service.process_file_upload(file)

@router.post("/upload_text", dependencies=[Depends(verify_api_key)])
async def upload_raw_text(request: RawTextUploadRequest):
    current_docs = rag_service.indexed_documents
    if not can_upload_doc(len(current_docs)):
        msg = "TRIAL_LIMIT_EXCEEDED: Your trial has expired. Activate a Lifetime License ($29) to index more documents." if len(current_docs) == 0 else "TRIAL_LIMIT_DOCS: Free evaluation trial allows indexing 1 document. Activate a Lifetime License ($29) to index unlimited documents."
        raise HTTPException(status_code=403, detail=msg)
    return await rag_service.process_raw_text_upload(request.title, request.content)

@router.delete("/document/{filename}", dependencies=[Depends(verify_api_key)])
async def delete_document(filename: str):
    return await rag_service.delete_single_document(filename)
