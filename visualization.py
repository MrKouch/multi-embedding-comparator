import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


class Visualizations:

    def plot_embeddings_2d(self, embedding_list, text_list):
        """
        Plot two embedding vectors in 2D using PCA.

        Args:
            embedding_list (list): List of embeddings
            text_list (list): Corresponding list of sentences

        Returns:
            Plotly Figure object
        """
        embeddings = np.array(embedding_list)
        labels = text_list

        # Reduce to 2D with PCA
        pca = PCA(n_components=len(embedding_list))
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
