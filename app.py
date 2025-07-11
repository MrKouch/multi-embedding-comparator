from sentence_transformers import SentenceTransformer
import streamlit as st

# 1. Load a pretrained Sentence Transformer model



st.title("🦜🔗 Text similarities")
first_sentence = st.text_input("First sentence", " ")
second_sentence = st.text_input("Second sentence", " ")
sentences = [first_sentence, second_sentence]

option = st.selectbox(
    "Select embedding",
    ("all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "all-distilroberta-v1",
    "paraphrase-MiniLM-L6-v2",
    "paraphrase-mpnet-base-v2",
    "nli-roberta-base-v2",
    "stsb-roberta-large",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "distiluse-base-multilingual-cased-v1",
    "sentence-transformers/LaBSE",
    "paraphrase-multilingual-mpnet-base-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "average_word_embeddings_glove.6B.300d",
    "msmarco-distilbert-base-v2",
    "instructor-xl",
    "e5-base",
    "bge-small-en-v1.5"))

option = st.selectbox(
    "Select distance metric",
    ())

if st.button("Run"):
    model = SentenceTransformer(option)
    embeddings = model.encode(sentences)
    # print(embeddings.shape)
    # [3, 384]
    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    distance = similarities[0][1]
    # print(similarities)

    st.write("The distance between the embeddings is ", distance.item())


