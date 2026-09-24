"""Joint query-passage cross-encoding; scores are relevance logits, not probabilities."""
from functools import lru_cache
try:
    from .config import SETTINGS
except ImportError:
    from config import SETTINGS

@lru_cache(maxsize=1)
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(SETTINGS.reranker_model,device='cpu',max_length=512)

def rerank(query,candidates,top_k=None):
    if not candidates:
        return []
    scores=load_reranker().predict([(query,row['text']) for row in candidates],show_progress_bar=False,batch_size=8)
    ranked=sorted([dict(row,reranker_score=float(score)) for row,score in zip(candidates,scores)],key=lambda r:r['reranker_score'],reverse=True)
    for n,row in enumerate(ranked,1):
        row['reranker_rank']=n
    return ranked if top_k is None else ranked[:top_k]
