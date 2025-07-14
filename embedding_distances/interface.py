from embedding_distances.distance_metrics import DistanceMetrics
import sentence_transformers
from sentence_transformers import SentenceTransformer


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
