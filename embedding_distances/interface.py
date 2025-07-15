from embedding_distances.distance_metrics import DistanceMetrics
import sentence_transformers
from sentence_transformers import SentenceTransformer
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


def calculate_distance_list(text_list, model_name="all-MiniLM-L6-v2", metric="cosine"):
    # Load embedding model
    model = SentenceTransformer(model_name)
    vecs = model.encode(text_list)

    # Calculate distance
    dm = DistanceMetrics()
    if not hasattr(dm, metric):
        raise ValueError(f"Unsupported metric: {metric}")
    distance_matrix = [[0 for i in range(len(text_list))] for j in range(len(text_list))]
    for i in range(len(text_list)):
        for j in range(len(text_list)):
            distance_matrix[i][j] = dm.calc_according_to_metric(metric, vecs[i], vecs[j])
    # distance = dm.calc_according_to_metric(metric, vec1, vec2)
    return distance_matrix

