import torch
from torch.nn import functional as F


def _masked_bce(logits, targets, available):
    available = available.bool()
    if not available.any(): return logits.sum()*0
    return F.binary_cross_entropy_with_logits(logits[available], targets[available])


def _multi_positive_nce(image, speech, ids, temperature=0.07):
    # Multiple spoken captions may correspond to the same image. Do NOT
    # penalise them as negatives as ordinary identity-diagonal CLIP does.
    logits = (image @ speech.T) / temperature
    positives = torch.tensor([[a == b for b in ids] for a in ids],device=image.device)
    def direction(sim, same):
        numerator = torch.logsumexp(sim.masked_fill(~same,-1e9),dim=-1)
        denominator = torch.logsumexp(sim,dim=-1)
        return (denominator-numerator).mean()
    return 0.5*(direction(logits,positives)+direction(logits.T,positives.T))


def prokg_loss(outputs,batch,method='full',weights=None):
    w = {'rec':1.,'ground':0.4,'verify':0.3,'cf':0.15,'ret':0.2,'graph':0.4}
    if weights: w.update(weights)
    attr = _masked_bce(outputs['attributes'],batch['attributes'],batch['has_attr'])
    rel = _masked_bce(outputs['relations'],batch['relations'],batch['has_rel'])
    cat = _masked_bce(outputs['categories'],batch['categories'],batch['has_cat'])
    rec = attr+rel+cat
    zero = rec*0
    graph_object,graph_attr,graph_rel = zero,zero,zero
    if method in ('progressive_kg','kg_refinement','full') and 'node_object_labels' in batch:
        obj_targets=batch['node_object_labels']
        obj_valid=obj_targets>=0
        if obj_valid.any():
            graph_object=F.cross_entropy(outputs['graph']['objects'][obj_valid],obj_targets[obj_valid])
        attr_targets=batch['node_attribute_labels']
        attr_valid=(attr_targets>=0).all(dim=-1)
        if attr_valid.any():
            graph_attr=F.binary_cross_entropy_with_logits(
                outputs['graph']['attributes'][attr_valid],attr_targets[attr_valid])
        edge_targets=batch['edge_relation_labels']
        edge_valid=edge_targets>=0
        if edge_valid.any():
            graph_rel=F.cross_entropy(outputs['graph']['edges'][edge_valid],edge_targets[edge_valid])
    graph_loss=graph_object+graph_attr+graph_rel
    ground,verify,cf,retrieval = zero,zero,zero,zero
    if method in ('progressive_kg','kg_refinement','full'):
        centers = batch['target_center']
        valid = (centers[:,:,0]>=0) & (batch['obs_mask']>0)
        if valid.any():
            grid = int(outputs['alignment'].shape[-1]**0.5)
            x = (centers[:,:,0].clamp(0,0.99999)*grid).long()
            y = (centers[:,:,1].clamp(0,0.99999)*grid).long()
            label = y*grid+x
            selected = outputs['alignment'][valid].clamp_min(1e-8)
            ground = F.nll_loss(selected.log(),label[valid])
    if method == 'full':
        labels = batch['evidence']
        valid = (labels>=0) & (batch['obs_mask']>0)
        if valid.any():
            verify = F.cross_entropy(outputs['evidence'][valid],labels[valid])
            # The image-only / prior prediction is the stop-gradient target.
            # This prevents contradicted and insufficient statements from
            # inducing a large output change without evidence.
            unreliable = valid & (labels!=0)
            if unreliable.any():
                old = outputs['stage_attributes'][:,:-1][unreliable].sigmoid().detach()
                new = outputs['stage_attributes'][:,1:][unreliable].sigmoid()
                old_rel = outputs['stage_relations'][:,:-1][unreliable].sigmoid().detach()
                new_rel = outputs['stage_relations'][:,1:][unreliable].sigmoid()
                cf = F.mse_loss(new,old)+F.mse_loss(new_rel,old_rel)
    if method != 'image_only':
        selected = batch['has_retrieval'].bool()
        if selected.sum()>1:
            ids = [key for key, keep in zip(batch['image_id'],selected.tolist()) if keep]
            retrieval = _multi_positive_nce(outputs['image_embedding'][selected],
                      outputs['speech_embedding'][selected],ids)
    total = w['rec']*rec+w['ground']*ground+w['verify']*verify+w['cf']*cf+w['ret']*retrieval+w['graph']*graph_loss
    components = {'total':total,'recognition':rec,'attribute':attr,'relationship':rel,
                  'multi_label':cat,'grounding':ground,'verification':verify,
                  'counterfactual':cf,'retrieval':retrieval,'graph':graph_loss}
    return total,components
