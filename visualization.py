import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from embedding_distances.distance_metrics import DistanceMetrics


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


def plot_embeddings_2d_with_distances(vectors, labels, highlight_index=None, distance_metric="cosine"):
    # Apply PCA
    dm = DistanceMetrics()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    df = pd.DataFrame(coords, columns=["x", "y"])
    df["label"] = labels

    fig = px.scatter(df, x="x", y="y", text="label")

    if highlight_index is not None:
        selected = df.iloc[highlight_index]
        selected_vec = coords[highlight_index]

        # Add red highlight for the selected point
        fig.add_trace(go.Scatter(
            x=[selected["x"]],
            y=[selected["y"]],
            mode="markers",
            marker=dict(color="red", size=12),
            name="Selected",
            showlegend=True
        ))

        # Add lines with distance labels
        for i, row in df.iterrows():
            if i == highlight_index:
                continue

            # Compute distance
            target_vec = coords[i]
            dist = dm.calc_according_to_metric(distance_metric, selected_vec, target_vec)

            # Midpoint for label
            mid_x = (selected["x"] + row["x"]) / 2
            mid_y = (selected["y"] + row["y"]) / 2

            # Line trace
            fig.add_trace(go.Scatter(
                x=[selected["x"], row["x"]],
                y=[selected["y"], row["y"]],
                mode="lines",
                line=dict(color="gray", width=1),
                showlegend=False,
                hoverinfo="skip"
            ))

            # Distance label trace (at midpoint)
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="text",
                text=[f"{dist:.2f}"],
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip"
            ))

    fig.update_traces(textposition='top center')
    fig.update_layout(title="2D PCA Embedding Visualization")
    return fig




###