from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass(frozen=True)
class Neo4jConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_db: str

@dataclass(frozen=True)
class KGRetrievalConfig(Neo4jConfig):
    embedding_model: str

class ConfigurationManager:
    def __init__(self):
        pass

    def get_neo4j_config(self) -> Neo4jConfig:
        config = Neo4jConfig(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_db=os.getenv("NEO4J_DB")
        )
        return config

    def get_kg_retrieval_config(self) -> KGRetrievalConfig:
        config = KGRetrievalConfig(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_db=os.getenv("NEO4J_DB"),
            embedding_model="thenlper/gte-base"
        )

        return config