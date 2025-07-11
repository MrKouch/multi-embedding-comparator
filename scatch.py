from sentence_transformers import SentenceTransformer
import streamlit as st

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")



st.title("🦜🔗 Text similarities")
first_sentence = st.text_input("First sentence", " ")
second_sentence = st.text_input("Second sentence", " ")
sentences = [first_sentence, second_sentence]


embeddings = model.encode(sentences)
# print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
distance = similarities[0][1]
# print(similarities)

st.write("The distance between the embeddings is ", distance)


