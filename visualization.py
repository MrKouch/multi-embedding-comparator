import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go


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

    def bars_graph(self, distance_tables, text_list):
        all_data = []

        for i, table in enumerate(distance_tables):
            distances = table.iloc[0].tolist()
            embedding_name = getattr(table, "name", f"Embedding {i+1}")  # Use name if available
            for text, dist in zip(text_list, distances):
                all_data.append({
                    'Text': text,
                    'Distance': dist,
                    'Embedding': embedding_name
                })

        df = pd.DataFrame(all_data)

        fig = px.bar(
            df,
            x='Text',
            y='Distance',
            color='Embedding',
            barmode='group',
            title='Distance from Anchor Sentence by Embedding Method'
        )

        fig.update_layout(xaxis_tickangle=-45)
        return fig


