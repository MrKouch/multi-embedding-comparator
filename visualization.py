import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

class Visualizations:
    def plot_embeddings_2d(self, embedding1, embedding2):
        """
        Plot two embedding vectors in 2D using PCA.

        Args:
            embedding1 (array-like): First embedding vector
            embedding2 (array-like): Second embedding vector

        Returns:
            Plotly Figure object
        """
        embeddings = np.array([embedding1, embedding2])
        labels = ["Sentence 1", "Sentence 2"]

        # Reduce to 2D with PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        # Create 2D scatter plot
        fig = px.scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            text=labels,
            labels={'x': 'PC 1', 'y': 'PC 2'},
            title="2D Projection of Embeddings (PCA)"
        )
        fig.update_traces(marker=dict(size=12))
        return fig
