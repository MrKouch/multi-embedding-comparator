from sentence_transformers import SentenceTransformer
import streamlit as st
from embedding_distances.distance_metrics import DistanceMetrics
import embedding_distances.interface as interface
from visualization import Visualizations  

st.title("Text similarities")

first_sentence = st.text_input("First sentence", " ")
second_sentence = st.text_input("Second sentence", " ")

embedding_type = st.selectbox(
    "Select embedding",
    (
        "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1",
        "paraphrase-MiniLM-L6-v2", "paraphrase-mpnet-base-v2", "nli-roberta-base-v2",
        "stsb-roberta-large", "paraphrase-multilingual-MiniLM-L12-v2",
        "distiluse-base-multilingual-cased-v1", "sentence-transformers/LaBSE",
        "paraphrase-multilingual-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1",
        "average_word_embeddings_glove.6B.300d", "msmarco-distilbert-base-v2",
        "instructor-xl", "e5-base", "bge-small-en-v1.5"
    )
)

distance_metric = st.selectbox(
    "Select distance metric",
    (
        "cosine", "euclidean", "dot", "manhattan",
        "chebyshev", "minkowski", "angular", "hamming", "jaccard"
    )
)

dm = DistanceMetrics()
viz = Visualizations()
# Add this ABOVE the Run button

show_plot = st.checkbox("Show embedding visualization (2D)")

if st.button("Run"):
    distance = interface.calculate_distance(first_sentence, second_sentence, embedding_type, distance_metric)
    st.write("The distance between the embeddings is", distance)

    # Only show plot if checkbox is checked
    model = SentenceTransformer(embedding_type)
    embeddings = model.encode([first_sentence, second_sentence])
    if show_plot:
        fig = viz.plot_embeddings_2d(embeddings[0], embeddings[1])
        st.plotly_chart(fig)

