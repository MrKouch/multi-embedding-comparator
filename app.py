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

embedding_class = st.selectbox(
    "Select embedding class",
    (
        "HFEmbeddingModel"
    )
)
models_list = interface.EMBEDDING_CLASSES[embedding_class].list_models()
# models_list = interface.EMBEDDING_CLASSES[embedding_class].models_list
embedding_type = st.selectbox(
    "Select embedding",
    models_list
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
    # embeddings = interface.encode_text_list(st.session_state.sentences, model_name=embedding_type)
    embeddings = interface.encode_text_list(text_list=st.session_state.sentences, embedding_class=embedding_class,
                                            model_id=embedding_type)
    unique_text_list = [f"{i + 1}. {s}" for i, s in enumerate(st.session_state.sentences)]
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
