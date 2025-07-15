from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding_distances import interface
import pandas as pd

app = FastAPI()

class DistanceRequest(BaseModel):
    sentences: list
    model: str = "all-MiniLM-L6-v2"
    distance_metric: str = "cosine"

@app.get("/")
def test():
    return {"hello": "world"}

@app.post("/dist")
def calculate_distance(req: DistanceRequest):
    if len(req.sentences) < 2:
        raise HTTPException(status_code=400, detail="At least two sentences are required.")
    embeddings_list = interface.encode_text_list(req.sentences, req.model)
    distance_matrix = interface.calculate_distance_list(
        text_list=req.sentences,
        embeddings_list=embeddings_list,
        metric=req.distance_metric
    )
    df = pd.DataFrame(distance_matrix, columns=req.sentences, index=req.sentences)
    return df.to_dict()

###