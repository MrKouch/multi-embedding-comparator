from embedding_distances.distance_metrics import DistanceMetrics
from sentence_transformers import SentenceTransformer
from embedding_distances.hf_model import HFEmbeddingModel
import numpy as np
import pandas as pd


def calculate_distance(text1, text2, model_name="all-MiniLM-L6-v2", metric="cosine"):
    # Load embedding model
    model = SentenceTransformer(model_name)
    vec1 = model.encode(text1)
    vec2 = model.encode(text2)

    # Calculate distance
    dm = DistanceMetrics()
    if not hasattr(dm, metric):
        raise ValueError(f"Unsupported metric: {metric}")
    distance = dm.calc_according_to_metric(metric, vec1, vec2)
    return distance


EMBEDDING_CLASSES = {
    "HFEmbeddingModel": HFEmbeddingModel,
    # "GeminiEmbeddingModel": GeminiEmbeddingModel,
    # "OpenAIEmbeddingModel": OpenAIEmbeddingModel,
    # "VertexAIEmbeddingModel": VertexAIEmbeddingModel,
    # Add others here
}


def encode_text_list(text_list, embedding_class="HFEmbeddingModel", model_id="all-MiniLM-L6-v2"):
    """
    Args:
        text_list (List[str]): Sentences to embed.
        embedding_class (str): Class name string from EMBEDDING_CLASSES.
        model_id (str): Model name or ID to pass to the constructor.

    Returns:
        List[List[float]]: Embedding vectors.
    """
    if embedding_class not in EMBEDDING_CLASSES:
        raise ValueError(f"Unsupported embedding class: {embedding_class}")

    model = EMBEDDING_CLASSES[embedding_class](model_id)
    return model.embed(text_list)


# def encode_text_list(text_list, model_name="all-MiniLM-L6-v2"):
#     model = SentenceTransformer(model_name)
#     vecs = model.encode(text_list)
#     return vecs


def calculate_distance_list(text_list, embeddings_list, metric="cosine"):
    dm = DistanceMetrics()
    if not hasattr(dm, metric):
        raise ValueError(f"Unsupported metric: {metric}")
    distance_matrix = [[0 for i in range(len(text_list))] for j in range(len(text_list))]
    for i in range(len(text_list)):
        for j in range(len(text_list)):
            distance_matrix[i][j] = dm.calc_according_to_metric(metric, embeddings_list[i], embeddings_list[j])
    return distance_matrix


# embed_list = encode_text_list(["text_list", "another text"], "HFEmbeddingModel", "all-MiniLM-L6-v2")
# embed_list = encode_text_list(["text_list", "another text"])
# print(calculate_distance_list(["text_list", "another text"], embed_list))
