from sentence_transformers import SentenceTransformer
import chromadb

class Embedder:
    def __init__(self, model_path):
        # This now correctly receives a local path like "./models/e5-small-v2"
        self.model = SentenceTransformer(model_path)
    
    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

class VectorIndex:
    def __init__(self, path, embedder):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection("docs")
        self.embedder = embedder

    def add(self, sections):
        texts = [s["text"] for s in sections]
        if not texts: return
        
        embs = self.embedder.encode(texts)
        ids = [s["id"] for s in sections]
        metas = [{"doc":s["doc"],"page":s["page"],"title":s["title"]} for s in sections]
        
        self.col.add(documents=texts, embeddings=embs.tolist(), ids=ids, metadatas=metas)

    def query(self, query_vec, n=10):
        # Ensure there are embeddings to query against
        if self.col.count() == 0:
            return {'ids': [[]], 'distances': [[]]}
        return self.col.query(query_embeddings=[query_vec.tolist()], n_results=min(n, self.col.count()))
