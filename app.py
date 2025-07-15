import streamlit as st
from embedding_distances.distance_metrics import DistanceMetrics
import embedding_distances.interface as interface
from visualization import Visualizations
import pandas as pd

st.title("Text similarities")

st.subheader("Sentence Collector")

# 1. Initialize list in session state
if "sentences" not in st.session_state:
    st.session_state.sentences = []

# 2. Input field for new sentence
new_sentence = st.text_input("Enter a new sentence:")

# 3. Add button to store the sentence
if st.button("Add Sentence"):
    if new_sentence.strip():  # Avoid empty inputs
        st.session_state.sentences.append(new_sentence.strip())
    else:
        st.warning("Please enter a valid sentence.")

# 4. Display current list
st.subheader("Collected Sentences:")
for idx, sentence in enumerate(st.session_state.sentences, 1):
    st.write(f"{idx}. {sentence}")


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

show_plot = st.checkbox("Show embedding visualization (2D)")

if st.button("Run"):
    embeddings = interface.encode_text_list(st.session_state.sentences, model_name=embedding_type)
    unique_text_list = [f"{i+1}. {s}" for i, s in enumerate(st.session_state.sentences)]
    distance_matrix = interface.calculate_distance_list(
        st.session_state.sentences,
        embeddings,
        distance_metric
    )

    distance_table = pd.DataFrame(distance_matrix,
                                  columns=unique_text_list,
                                  index=unique_text_list)

    st.table(distance_table)

    # Only show plot if checkbox is checked
    if show_plot:
        fig = viz.plot_embeddings_2d(list(embeddings), st.session_state.sentences)
        st.plotly_chart(fig)
