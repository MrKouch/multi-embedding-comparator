from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from distance_metrics import DistanceMetrics
from pydantic import BaseModel


fastAPI = FastAPI()

@fastAPI.get("/")
def test():
    return {"hello": "world"}

class DistanceRequest(BaseModel):
    sentence1: str
    sentence2: str
    model_choice: str = "all-MiniLM-L6-v2"
    distance_metric: str = "cosine"

@fastAPI.post("/dist")
def calculate_distance(req: DistanceRequest):
    sentences = [req.sentence1, req.sentence2]
    model = SentenceTransformer(model_name_or_path=req.model_choice)
    dm = DistanceMetrics()
    embeddings = model.encode(sentences)
    distance = dm.calc_according_to_metric(req.distance_metric, embeddings[0], embeddings[1])
    
    return {
        "sentence1": req.sentence1,
        "sentence2": req.sentence2,
        "model_choice": req.model_choice,
        "distance_metric": req.distance_metric,
        "distance": distance
    }




