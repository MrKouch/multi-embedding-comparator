import streamlit as st
from embedding_distances.distance_metrics import DistanceMetrics
import embedding_distances.interface as interface
from embedding_distances.visualization import Visualizations
import pandas as pd
from embedding_distances import visualization

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
# st.subheader("Collected Sentences:")
# for idx, sentence in enumerate(st.session_state.sentences, 1):
#     st.write(f"{idx}. {sentence}")

# 4. Display current list with delete buttons
st.subheader("Collected Sentences:")
for idx, sentence in enumerate(st.session_state.sentences):
    col1, col2 = st.columns([8, 1])
    with col1:
        st.write(f"{idx + 1}. {sentence}")
    with col2:
        if st.button("❌", key=f"delete_{idx}"):
            st.session_state.sentences.pop(idx)
            st.rerun()

embedding_class = st.selectbox(
    "Select embedding class",
    (
        # "HFEmbeddingModel"
        interface.EMBEDDING_CLASSES.keys()
    )
)
models_list = interface.EMBEDDING_CLASSES[embedding_class].list_models()
# models_list = interface.EMBEDDING_CLASSES[embedding_class].models_list
# embedding_type = st.selectbox(
#     "Select embedding",
#     models_list
# )
embedding_options = st.multiselect(
    "Select desired embedding options",
    models_list,
    default="all-MiniLM-L6-v2"
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

show_distance_table = st.checkbox("Show distance table",
                                  help="Table that shows the distance itself between every two sentences")
show_plot = st.checkbox("Show embedding visualization (2D)",
                        help="2D PCA of the sentences, "
                             "shown with the original distance between the reference sentence to the others. "
                             "shows one graph for each chosen embedding type")
show_bars = st.checkbox("Show bars visualization (2D)",
                        help="Shows the distance between the reference sentence to all the others "
                             "using bars graph. shows one graph for each chosen embedding type")

selected_index = st.selectbox(
    "Select a reference sentence to compare distances against:",
    options=list(range(len(st.session_state.sentences))),
    format_func=lambda i: f"{i + 1}. {st.session_state.sentences[i]}",
    help="The sentence from which the distances to all other sentences will be measured "
)

labels = [str(i + 1) for i in range(len(st.session_state.sentences))]
if st.button("Run"):
    tables = []
    embeddings = []
    for embedding_option in embedding_options:
        # embeddings = interface.encode_text_list(st.session_state.sentences, model_name=embedding_type)
        embedding = interface.encode_text_list(text_list=st.session_state.sentences, embedding_class=embedding_class,
                                               model_id=embedding_option)
        embeddings.append(embedding)
        unique_text_list = [f"{i + 1}. {s}" for i, s in enumerate(st.session_state.sentences)]
        distance_matrix = interface.calculate_distance_list(
            st.session_state.sentences,
            embedding,
            distance_metric
        )

        distance_table = pd.DataFrame(distance_matrix,
                                      columns=labels,
                                      index=labels)
        if show_distance_table or show_plot:
            st.subheader(f"{embedding_option}")
        if show_distance_table:
            table = st.table(distance_table)
        tables.append(distance_table)
        # Only show plot if checkbox is checked
        if show_plot:

            fig = visualization.plot_embeddings_2d_with_distances(embedding, labels,
                                                                  highlight_index=selected_index,
                                                                  hover_texts=st.session_state.sentences)
            st.plotly_chart(fig)
            # clicked_point = plotly_events(fig, click_event=True, select_event=False, hover_event=False,
            #                               key=embedding_option)
            # if clicked_point:
            #     selected_index = int(clicked_point[0]['pointIndex'])
            #     st.session_state.selected_index = selected_index
            #     st.success(f"Selected sentence {selected_index + 1} as reference.")

    if show_bars:
        fig = viz.bars_graph(distance_tables=tables, embedding_names=embedding_options,
                             text_list=labels, anchor_index=selected_index)
        st.plotly_chart(fig)

###
