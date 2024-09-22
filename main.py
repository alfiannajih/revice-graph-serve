from transformers import pipeline
import torch
import os
from fastapi import FastAPI, Request

from configuration import ConfigurationManager
from kg_retrieval import Neo4JConnection, KnowledgeGraphRetrievalPipeline

pipe = pipeline(
    "g-retriever-task",
    model=f"{os.environ['AIP_STORAGE_URI']}/g-retriever",
    torch_dtype=torch.int8,
    trust_remote_code=True
)
config = ConfigurationManager()

neo4j_config = config.get_neo4j_config()
neo4j_connection = Neo4JConnection(neo4j_config)

kg_retrieval_config = config.get_kg_retrieval_config()
kg_retrieval_pipeline = KnowledgeGraphRetrievalPipeline(kg_retrieval_config, neo4j_connection)

HEALTH_ROUTE = os.environ["AIP_HEALTH_ROUTE"]
PREDICTIONS_ROUTE = os.environ["AIP_PREDICT_ROUTE"]

app = FastAPI()

# class Resume(BaseModel):
#     education: str
#     experience: str
#     project: str
#     skill: str
#     description: str

#     model_config = {
#         "json_schema_extra": {
#             "examples": [
#                 {
#                     "education": "bachelor degree of mathematics",
#                     "experience": "data scientist intern at banking industry",
#                     "project": "churn prediction",
#                     "skill":  "python, pytorch",
#                     "description": "I want to pursue my career in machine learning engineer",
#                 }
#             ]
#         }
#     }

# class Generation(BaseModel):
#     max_new_tokens: int
#     temperature: float
#     top_p: float
#     do_sample: bool

#     model_config = {
#         "json_schema_extra": {
#             "examples": [
#                 {
#                     "max_new_tokens": 512,
#                     "temperature": 1,
#                     "top_p": 0.9,
#                     "do_sample": True
#                 }
#             ]
#         }
#     }

# class Prompt(BaseModel):
#     resume: Resume
#     generation: Generation

# class Review(BaseModel):
#     review: str

@app.get(HEALTH_ROUTE, status_code=200)
def health():
    return {"Healthy Server!"}

@app.post(PREDICTIONS_ROUTE)
async def predict(request: Request):
    # input json is the Request and will be read into `instances`
    body = await request.json()
    content = body["instances"][0]
    raw_resume = content["resume"]
    generate_kwargs = content["generation"]

    resume_component = [
        raw_resume["education"],
        raw_resume["experience"],
        raw_resume["project"],
        raw_resume["skill"]
    ]
    graph, textualized_graph = retrieve_kg(resume_component, raw_resume["description"])
    resume = create_resume(raw_resume["education"], raw_resume["experience"], raw_resume["project"], raw_resume["skill"])

    raw_resume = resume + "\n\n" + "DESCRIPTION\n{}".format(raw_resume["description"])
    inputs = {
        "inputs": resume,
        "textualized_graph": textualized_graph,
        "graph": graph
    }

    review = pipe(
        generate_kwargs=generate_kwargs,
        **inputs
    )

    return {"predictions": review}

def create_resume(
        education, 
        experience, 
        project, 
        skill
    ):
    return "RESUME\neducation\n{}experience\n{}\nproject\n{}\nskills\n{}".\
        format(education, experience, project, skill)

def retrieve_kg(
        resume_component,
        description
    ):
    subgraph, textualized_graph = kg_retrieval_pipeline.graph_retrieval_pipeline(resume_component, description)
    
    return subgraph, textualized_graph