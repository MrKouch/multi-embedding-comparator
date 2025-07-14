import langchain.embeddings as lng
from embedding_distances.distance_metrics import DistanceMetrics

# this class is intended to have the logic, s.t. the web interface, API and python library can use it
# getatr - maybe use
class EmbeddingEngine:

    def __init__(self, model: str, provider: str = None, distance_metric: str = "cosine", api_key: str = None):
        self.distance_mtric = distance_metric
        self.provider = provider
        self.model = self._load_embedding_model()

    # only a few providers from the Langchain.embeddings options. can add more
    def _load_embedding_model(self):
        if self.provider == "openai":
            return lng.OpenAIEmbeddings(model=self.model_name, openai_api_key=self.api_key)
        elif self.provider == "huggingface":
            return lng.HuggingFaceEmbeddings(model_name=self.model_name)
        elif self.provider == "cohere":
            return lng.CohereEmbeddings(model=self.model_name, cohere_api_key=self.api_key)
        elif self.provider == "vertexai":
            return lng.VertexAIEmbeddings(model_name=self.model_name)
        elif self.provider == "ollama":
            return lng.OllamaEmbeddings(model=self.model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
    def calculate_distance(self, text1: str, text2: str):
        vectors = self.embedding_model.embed_documents([text1, text2])
        dm = DistanceMetrics()
        return dm.calc_according_to_metric(self.distance_metric, vectors[0], vectors[1])
        