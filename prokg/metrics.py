import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def multilabel_macro_f1(labels,logits,threshold=0.5):
    y = np.asarray(labels, dtype=int)
    s = np.asarray(logits)
    if y.size==0: return None
    return float(100*f1_score(y,(1/(1+np.exp(-np.clip(s,-50,50))))>=threshold,
                               average='macro',zero_division=0))


def multilabel_map(labels,logits):
    y = np.asarray(labels,dtype=int)
    s = np.asarray(logits)
    if y.size==0: return None
    present = [(y[:,i].sum()>0 and y[:,i].sum()<len(y)) for i in range(y.shape[1])]
    if not any(present): return None
    return float(100*np.mean([average_precision_score(y[:,i],s[:,i])
                              for i,ok in enumerate(present) if ok]))


def recall_at_k(image_embeddings,speech_embeddings,image_ids,k=10):
    """Deduplicated image gallery, spoken queries. Multiple captions/image allowed."""
    image = np.asarray(image_embeddings,dtype=np.float32)
    speech = np.asarray(speech_embeddings,dtype=np.float32)
    gallery = {}
    for emb,key in zip(image,image_ids):
        if key not in gallery: gallery[key] = emb
    candidates = np.stack(list(gallery.values()))
    candidate_ids = np.array(list(gallery.keys()))
    if not len(speech): return None
    # Query in blocks to avoid a 50k x 50k similarity matrix on SON.
    correct=0
    kk=min(k,len(candidates))
    for start in range(0,len(speech),128):
        scores=speech[start:start+128] @ candidates.T
        chosen=np.argpartition(scores,kth=len(candidates)-kk,axis=1)[:,-kk:]
        for offset,top in enumerate(chosen):
            correct+=image_ids[start+offset] in candidate_ids[top]
    return float(100*correct/len(speech))
