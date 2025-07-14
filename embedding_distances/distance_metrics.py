import torch
import torch.nn.functional as F
import numpy as np


class DistanceMetrics:

    @staticmethod
    def _prepare(vec1, vec2):
        # Convert NumPy arrays to torch tensors
        if isinstance(vec1, np.ndarray):
            vec1 = torch.tensor(vec1, dtype=torch.float32)
        if isinstance(vec2, np.ndarray):
            vec2 = torch.tensor(vec2, dtype=torch.float32)

        # Ensure tensors are 2D
        if vec1.dim() == 1:
            vec1 = vec1.unsqueeze(0)
        if vec2.dim() == 1:
            vec2 = vec2.unsqueeze(0)

        return vec1, vec2

    def cosine(self, vec1, vec2):
        vec1, vec2 = self._prepare(vec1, vec2)
        distance = 1 - F.cosine_similarity(vec1, vec2).item()
        return round(distance, 6)

    def euclidean(self, vec1, vec2):
        vec1, vec2 = self._prepare(vec1, vec2)
        distance = F.pairwise_distance(vec1, vec2, p=2).item()
        return round(distance, 6)

    def manhattan(self, vec1, vec2):
        vec1, vec2 = self._prepare(vec1, vec2)
        distance = F.pairwise_distance(vec1, vec2, p=1).item()
        return round(distance, 6)

    def dot(self, vec1, vec2):
        vec1, vec2 = self._prepare(vec1, vec2)
        distance = torch.dot(vec1.flatten(), vec2.flatten()).item()
        return round(distance, 6)

    def chebyshev(self, vec1, vec2):
        vec1, vec2 = self._prepare(vec1, vec2)
        distance = torch.max(torch.abs(vec1 - vec2)).item()
        return round(distance, 6)

    def minkowski(self, vec1, vec2, p=3):  # p can be adjusted as needed
        vec1, vec2 = self._prepare(vec1, vec2)
        distance = F.pairwise_distance(vec1, vec2, p=p).item()
        return round(distance, 6)

    def angular(self, vec1, vec2):
        # Angular distance = arccos(similarity) / π
        vec1, vec2 = self._prepare(vec1, vec2)
        sim = F.cosine_similarity(vec1, vec2).clamp(-1 + 1e-7, 1 - 1e-7)
        distance = (torch.acos(sim) / torch.pi).item()
        return round(distance, 6)

    def hamming(self, vec1, vec2):
        # Assumes binary vectors (0/1), or thresholded
        vec1, vec2 = self._prepare(vec1, vec2)
        distance = torch.mean((vec1 != vec2).float()).item()
        return round(distance, 6)

    def jaccard(self, vec1, vec2):
        # Assumes binary vectors (0/1)
        vec1, vec2 = self._prepare(vec1, vec2)
        intersection = torch.sum((vec1.bool() & vec2.bool()).float())
        union = torch.sum((vec1.bool() | vec2.bool()).float())
        if union == 0:
            return 0.0
        distance = 1 - (intersection / union).item()
        return round(distance, 6)

    def calc_according_to_metric(self, metric, vec1, vec2):
        if not hasattr(self, metric):
            raise ValueError(f"Unknown distance metric: {metric}")

        method = getattr(self, metric)

        # Handle special case for Minkowski (needs `p`)
        if metric == "minkowski":
            return method(vec1, vec2, p=3)

        return method(vec1, vec2)

