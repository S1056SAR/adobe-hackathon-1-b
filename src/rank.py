# src/rank.py

import numpy as np
from rank_bm25 import BM25Okapi
from src.embed import Embedder # Import the Embedder class

def rank_sections(persona, job, sections, vectordb, embedder_model_name):
    # embed persona+job
    embedder = Embedder(embedder_model_name)
    qtext = persona + " " + job
    qvec  = embedder.encode([qtext])[0]
    
    # ANN search
    # Ensure the query returns results before proceeding
    results = vectordb.query(qvec, n=30)
    if not results or not results.get('ids') or not results['ids'][0]:
        return []

    res_ids = results['ids'][0]
    res_distances = results['distances'][0]

    # BM25 over titles
    corpus = [s["title"].split() for s in sections]
    if not corpus: return [] # Avoid error on empty corpus
    bm25 = BM25Okapi(corpus)
    
    scores = []
    # Use a dictionary for quick section lookups
    sections_dict = {s["id"]: s for s in sections}

    for idx, doc_id in enumerate(res_ids):
        sec = sections_dict.get(doc_id)
        if not sec: continue

        dense = 1 - res_distances[idx] # Convert distance to similarity
        
        # Tokenize query for BM25
        tokenized_query = qtext.split()
        bm_score_list = bm25.get_scores(tokenized_query)
        
        # Find the index of the current section in the original list to get its BM25 score
        try:
            sec_index = list(sections_dict.keys()).index(doc_id)
            bm = bm_score_list[sec_index]
        except (ValueError, IndexError):
            bm = 0.0
            
        # The cross-encoder was too complex, use a simple weighted score for now
        # Re-introduce cross-encoder if time permits and it meets constraints
        score = 0.7 * dense + 0.3 * bm
        scores.append((score, sec))
        
    scores.sort(key=lambda x:-x[0])
    ranked = []
    for rank, (_, sec) in enumerate(scores[:10], start=1):
        sec["rank"] = rank
        ranked.append(sec)
    return ranked
