"""BGE-M3 + the original FAISS index. The original retrieve(query, top_k) API is preserved."""
import json,re
from functools import lru_cache
try:
    from .config import ROOT, SETTINGS
except ImportError:
    from config import ROOT, SETTINGS

BASE_DIR = ROOT
VECTORSTORE_DIR = ROOT/'vectorstore'
INDEX_PATH = VECTORSTORE_DIR/'index.faiss'
METADATA_PATH = VECTORSTORE_DIR/'metadata.json'

@lru_cache(maxsize=1)
def load_store():
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    if index.ntotal != len(metadata) or index.d != 1024 or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError('Existing index/metadata mismatch; index was NOT changed.')
    required={'text','source','page','document_id','parent_id','chunk_id','parent_text'}
    if any(not required.issubset(x) for x in metadata):
        raise ValueError('Missing hierarchical metadata')
    return index, metadata

@lru_cache(maxsize=1)
def load_embedding_model():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(SETTINGS.cpu_threads)
    model = SentenceTransformer(SETTINGS.embedding_model, device=SETTINGS.embedding_device)
    if model.get_embedding_dimension() != 1024:
        raise ValueError('Embedding model incompatible with existing index')
    return model

def retrieve(query, top_k=5):
    import numpy as np
    if not isinstance(query,str) or not query.strip():
        raise ValueError('Question must not be empty')
    if top_k < 1:
        raise ValueError('top_k must be positive')
    index, metadata = load_store()
    vector = np.asarray(load_embedding_model().encode([query],normalize_embeddings=True,show_progress_bar=False),dtype='float32')
    scores, indices = index.search(vector, min(top_k,index.ntotal))
    return [dict(metadata[int(i)],child_id=metadata[int(i)]['chunk_id'],similarity_score=float(s),retrieval_rank=rank)
            for rank,(s,i) in enumerate(zip(scores[0],indices[0]),1) if i >= 0]

def focused_query(query):
    """Domain-neutral comparison templates; no source names, facts, or expected answers."""
    if not SETTINGS.comparison_query_focus: return query
    patterns=[r'^does\s+.+?\s+agree\s+with\s+.+?\s+(?:about|regarding|on)\s+(.+?)[?.!]*$',
              r'^compare\s+.+?\s+(?:for|regarding|in terms of)\s+(.+?)[?.!]*$']
    for pattern in patterns:
        match=re.match(pattern,query.strip(),flags=re.IGNORECASE)
        if match and len(match.group(1).split())>=2:
            return 'What was the reported '+match.group(1).strip(' ?.！')+'?'
    return query

def retrieve_candidates(query,top_k):
    focus=focused_query(query)
    variants=[query] if focus==query else [query,focus]
    combined={}
    for variant in variants:
        for row in retrieve(variant,top_k):
            entry=combined.setdefault(row['chunk_id'],row)
            entry.setdefault('retrieval_queries',[]).append({'query':variant,'rank':row['retrieval_rank'],'score':row['similarity_score']})
            # Query variants' scores are not calibrated against one another; retain first-seen order.
            entry['reranking_query']=focus
    return list(combined.values())

if __name__ == '__main__':
    import argparse,sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('query',nargs='?')
    parser.add_argument('--top-k',type=int,default=5)
    args=parser.parse_args()
    query=args.query or input('Enter your question: ')
    for n,row in enumerate(retrieve(query,args.top_k),1):
        print(f"\nRESULT {n} | {row['source']} | page {row['page']} | {row['chunk_id']} | score {row['similarity_score']:.4f}\n{row['text']}\nParent ID: {row['parent_id']}")
