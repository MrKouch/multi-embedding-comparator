# embeddings/hf_model.py
from sentence_transformers import SentenceTransformer
from .base import EmbeddingModel


class HFEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self._name = model_name

    def embed(self, sentences):
        return self.model.encode(sentences)#, convert_to_numpy=True)

    def name(self):
        return self._name

    @classmethod
    def list_models(cls):
        return ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1",
                "paraphrase-MiniLM-L6-v2", "paraphrase-mpnet-base-v2", "nli-roberta-base-v2",
                "stsb-roberta-large", "paraphrase-multilingual-MiniLM-L12-v2",
                "distiluse-base-multilingual-cased-v1", "sentence-transformers/LaBSE",
                "paraphrase-multilingual-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1",
                "average_word_embeddings_glove.6B.300d", "msmarco-distilbert-base-v2",
                "instructor-xl", "e5-base", "bge-small-en-v1.5"]
