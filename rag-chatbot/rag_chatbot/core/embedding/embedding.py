import os
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel, AutoTokenizer
from ...setting import RAGSettings
from dotenv import load_dotenv

load_dotenv()


class LocalEmbedding:
    @staticmethod
    def set(setting: RAGSettings | None = None, **kwargs):
        setting = setting or RAGSettings()
        model_name = setting.ingestion.embed_llm

        return HuggingFaceEmbedding(
            model=AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ),
            cache_folder=os.path.join(os.getcwd(), setting.ingestion.cache_folder),
            trust_remote_code=True,
            embed_batch_size=setting.ingestion.embed_batch_size
        )
