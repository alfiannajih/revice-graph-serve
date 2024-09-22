from neo4j import GraphDatabase
import pandas as pd
import torch
from torch_geometric.data import Data

from configuration import Neo4jConfig, KGRetrievalConfig
from utils import get_emb_model, retrieval_via_pcst

class Neo4JConnection:
    def __init__(self, config: Neo4jConfig):
        self.driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))
        self.db = config.neo4j_db
        self.driver.verify_connectivity()

    def get_session(self):
        return self.driver.session(database=self.db)

    def get_head_node(self, relation_ids):
        node_ids = self.driver.execute_query(
            """
            MATCH (h)-[r]->()
            WHERE elementId(r) IN {}
            RETURN DISTINCT elementId(h) AS id
            """.format(relation_ids)
        )
        nodes = [node.value() for node in node_ids.records]
        
        return nodes
    
    def get_tail_node(self, relation_ids):
        node_ids = self.driver.execute_query(
            """
            MATCH ()-[r]->(t)
            WHERE elementId(r) IN {}
            RETURN DISTINCT elementId(t) AS id
            """.format(relation_ids)
        )
        nodes = [node.value() for node in node_ids.records]
        
        return nodes
    
    def get_tail_connection_from_head(self, head_ids):
        relation_ids = self.driver.execute_query(
            """
            MATCH (h)-[r]->()
            WHERE elementId(h) IN {}
            RETURN DISTINCT elementId(r) AS id LIMIT 50
            """.format(head_ids)
        )
        relations = [relation.value() for relation in relation_ids.records]

        return relations

    def close(self):
        self.driver.close()

class KnowledgeGraphRetrieval:
    def __init__(self, config: KGRetrievalConfig, neo4j_connection: Neo4JConnection):
        self.config = config
        self.neo4j_connection = neo4j_connection
        
        self.embedding_model = get_emb_model(self.config.embedding_model)

    def query_relationship_from_node(self, query, n_query):
        similar_relations = self.neo4j_connection.driver.execute_query(
            """
            CALL db.index.vector.queryNodes('JobTitleIndex', {}, {})
            YIELD node, score
            MATCH p=(node)-[r:offered_by]->(connectedNode)
            RETURN elementId(r) AS id, r.description, r.location
            """.format(n_query, query)
        )
        
        relations = []
        for relation in similar_relations.records:
            _id = relation.get("id")
            text = "Job description: {}".format(relation.get("r.description"), relation.get("r.location"))

            relations.append({"rel_id": _id, "text": text})
        
        return relations
    
class KnowledgeGraphRetrievalPipeline(KnowledgeGraphRetrieval):
    def __init__(
        self,
        config: KGRetrievalConfig,
        neo4j_connection: Neo4JConnection
    ):
        KnowledgeGraphRetrieval.__init__(self, config, neo4j_connection)

    def triples_retrieval(self, resume, desc, top_emb=5):
        query = resume + [desc]
        query_emb = self.embedding_model.encode(query, show_progress_bar=False).mean(axis=0).tolist()

        relations = self.query_relationship_from_node(query_emb, top_emb)
        relation_ids = [r["rel_id"] for r in relations]

        tail_ids = self.neo4j_connection.get_tail_node(relation_ids)
        tail_connection = self.neo4j_connection.get_tail_connection_from_head(tail_ids)
        
        head_ids = self.neo4j_connection.get_head_node(relation_ids)
        head_connection = self.neo4j_connection.get_tail_connection_from_head(head_ids)

        return relation_ids + tail_connection + head_connection, torch.tensor(query_emb)
    
    def build_graph(self, triples, query_emb):
        with self.neo4j_connection.get_session() as session:
            result = session.run(
                """
                MATCH (h)-[r]->(t)
                WHERE elementId(r) IN {}
                RETURN h.name AS h_name, h.embedding AS h_embedding, TYPE(r) AS r_type, r.embedding AS r_embedding, r.description AS job_description, t.embedding AS t_embedding, t.name AS t_name
                """.format(triples)
            )

            head_nodes = []
            tail_nodes = []
            node_embedding = []
            node_mapping = {}
            edge_attr = []
            edges = []
            nodes = {}

            for rec in result:
                if rec.get("h_name") not in node_mapping:
                    node_embedding.append(rec.get("h_embedding"))
                    nodes[len(node_mapping)] = rec.get("h_name")
                    node_mapping[rec.get("h_name")] = len(node_mapping)

                if rec.get("t_name") not in node_mapping:
                    node_embedding.append(rec.get("t_embedding"))
                    nodes[len(node_mapping)] = rec.get("t_name")
                    node_mapping[rec.get("t_name")] = len(node_mapping)

                head_nodes.append(rec.get("h_name"))
                tail_nodes.append(rec.get("t_name"))
                edge_attr.append(rec.get("r_embedding"))

                if rec.get("job_description") != None:
                    textualized_prop = "{}\nJob Description: {}".format(rec.get("r_type"), rec.get("job_description"))
                else:
                    textualized_prop = rec.get("r_type")

                edges.append({
                    "src": node_mapping[rec.get("h_name")],
                    "edge_attr": textualized_prop,
                    "dst": node_mapping[rec.get("t_name")]
                })
            
            src = [node_mapping[index] for index in head_nodes]
            dst = [node_mapping[index] for index in tail_nodes]

            edge_index = torch.tensor([src, dst])
            edge_attr = torch.tensor(edge_attr)

            graph = Data(x=torch.tensor(node_embedding), edge_index=edge_index, edge_attr=edge_attr)
            nodes = pd.DataFrame([{'node_id': k, 'node_attr': v} for k, v in nodes.items()], columns=['node_id', 'node_attr'])
            edges = pd.DataFrame(edges, columns=['src', 'edge_attr', 'dst'])
            
            subgraph, desc = retrieval_via_pcst(graph, query_emb, nodes, edges, topk=10, topk_e=3, cost_e=0.5)

            return subgraph, desc
    
    def graph_retrieval_pipeline(self, resume, desc, top_emb=5):
        triples, query_emb = self.triples_retrieval(resume, desc, top_emb)
        subgraph, textualize_graph = self.build_graph(triples, query_emb)
        
        return subgraph, textualize_graph