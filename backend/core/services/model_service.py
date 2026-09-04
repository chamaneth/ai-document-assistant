from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from core.config import settings

_embeddings = None
_llm = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print(f"[Model Service] Loading Embeddings ({settings.EMBEDDING_MODEL_NAME})...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            cache_folder=settings.EMBEDDINGS_DIR
        )
    return _embeddings

def get_llm():
    global _llm
    if _llm is None:
        print(f"[Model Service] Loading LLM Pipeline ({settings.LLM_MODEL_NAME})...")
        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME, cache_dir=settings.HUB_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.LLM_MODEL_NAME, cache_dir=settings.HUB_DIR)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=False
        )
        _llm = HuggingFacePipeline(pipeline=pipe)
    return _llm
