import torch
from torch import nn
from torch.nn import functional as F

METHODS = ('image_only', 'direct_fusion', 'progressive_fusion',
           'progressive_kg', 'kg_refinement', 'full')


class SmallVisionEncoder(nn.Module):
    def __init__(self, dim=64, grid=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32,48,3,stride=2,padding=1), nn.BatchNorm2d(48), nn.GELU(),
            nn.Conv2d(48,dim,3,stride=2,padding=1), nn.BatchNorm2d(dim), nn.GELU(),
            nn.AdaptiveAvgPool2d((grid,grid)))
        self.grid = grid

    def forward(self, image):
        return self.cnn(image).flatten(2).transpose(1,2)


class CompactSpeechEncoder(nn.Module):
    """Acoustic log-mel CNN + supplied transcript token embedding (ASR not included)."""
    def __init__(self, dim=64, sample_rate=16000):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=32)
        self.audio_conv = nn.Sequential(nn.Conv2d(1,12,3,padding=1),nn.GELU(),
                          nn.MaxPool2d(2),nn.Conv2d(12,24,3,padding=1),nn.GELU(),
                          nn.AdaptiveAvgPool2d((1,1)))
        self.text = nn.Embedding(8192,dim,padding_idx=0)
        self.fuse = nn.Sequential(nn.Linear(24+dim,dim),nn.LayerNorm(dim),nn.GELU())

    def forward(self, waveform, tokens):
        # waveform B,T,L ; tokens B,T,W
        b,t,length = waveform.shape
        mel = torch.log1p(self.mel(waveform.reshape(b*t,length))).unsqueeze(1)
        audio_feats = self.audio_conv(mel).flatten(1)
        tokens = tokens.reshape(b*t,-1)
        token_emb = self.text(tokens)
        valid = (tokens != 0).unsqueeze(-1)
        text_feats = (token_emb * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.fuse(torch.cat([audio_feats,text_feats],dim=-1)).reshape(b,t,-1)


class ProKGVision(nn.Module):
    def __init__(self, num_objects, num_attributes, num_relations, num_categories,
                 method='full', dim=64, grid=4, topk=3, threshold=0.20):
        super().__init__()
        if method not in METHODS: raise ValueError(f'Unknown method: {method}')
        self.method = method
        self.dim, self.grid, self.topk, self.threshold = dim,grid,topk,threshold
        self.visual = SmallVisionEncoder(dim,grid)
        self.speech = CompactSpeechEncoder(dim)
        self.graph_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.graph_norm = nn.LayerNorm(dim)
        self.objects = nn.Linear(dim,num_objects)
        self.node_attributes = nn.Linear(dim,num_attributes)
        self.edges = nn.Sequential(nn.Linear(2*dim,dim),nn.GELU(),nn.Linear(dim,num_relations))
        self.proj_obs, self.proj_graph = nn.Linear(dim,dim),nn.Linear(dim,dim)
        self.simple_update = nn.Linear(dim,dim)
        self.refiner = nn.Sequential(nn.Linear(dim*3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.verify = nn.Sequential(nn.Linear(dim*3,dim),nn.GELU(),nn.Linear(dim,3))
        self.gate = nn.Linear(2*dim+3,1)
        self.attr_head = nn.Linear(dim,num_attributes)
        self.rel_head = nn.Linear(dim,num_relations)
        self.cat_head = nn.Linear(dim,num_categories)
        self.image_projection = nn.Linear(dim,dim)
        self.audio_projection = nn.Linear(dim,dim)

    def graph(self, f):
        message,_ = self.graph_attn(f,f,f,need_weights=False)
        h = self.graph_norm(f + message)
        b,n,d = h.shape
        left = h.unsqueeze(2).expand(-1,-1,n,-1)
        right = h.unsqueeze(1).expand(-1,n,-1,-1)
        edge_logits = self.edges(torch.cat([left,right],dim=-1))
        return {'nodes':h, 'objects':self.objects(h),
                'attributes':self.node_attributes(h), 'edges':edge_logits}

    def heads(self, f):
        pooled = f.mean(dim=1)
        return (self.attr_head(pooled),self.rel_head(pooled),self.cat_head(pooled))

    def forward(self, image, audio, tokens, obs_mask):
        f = self.visual(image)
        graph = self.graph(f)  # No target boxes or annotated nodes are passed here.
        base = f.clone()
        b,t = obs_mask.shape
        obs = self.speech(audio,tokens)
        stages = [self.heads(f)]
        alignment_list, evidence_list, gates, masks = [],[],[],[]
        for step in range(t):
            active = obs_mask[:,step].view(b,1,1)
            o = obs[:,step]
            attention = torch.zeros(b,f.shape[1],device=f.device)
            region_mask = torch.zeros_like(attention)
            evidence = torch.zeros(b,3,device=f.device)
            gate = torch.zeros(b,device=f.device)
            if self.method == 'image_only':
                pass
            elif self.method == 'direct_fusion':
                # All observations are fused in a single update below.
                pass
            elif self.method == 'progressive_fusion':
                delta = torch.tanh(self.simple_update(o)).unsqueeze(1)
                f = f + active * delta
            else:
                node_embeddings = graph['nodes']
                sim = F.cosine_similarity(self.proj_obs(o).unsqueeze(1),
                                          self.proj_graph(node_embeddings),dim=-1)
                # Relation connectivity contributes to graph-guided alignment.
                connectivity = graph['edges'].sigmoid().amax(-1).mean(1)
                scores = sim + 0.10*connectivity
                attention = torch.softmax(scores,dim=-1)
                k = min(self.topk,attention.shape[-1])
                _,indices = attention.topk(k,dim=-1)
                region_mask = torch.zeros_like(attention).scatter(1,indices,1.)
                region_mask = region_mask * attention * attention.shape[1]/k
                q = sim.max(dim=-1).values
                is_aligned = (q >= self.threshold).float().view(b,1,1)
                region_mask = region_mask * is_aligned.squeeze(-1)
                if self.method == 'progressive_kg':
                    delta = self.simple_update(o).unsqueeze(1)
                    f = f + active*region_mask.unsqueeze(-1)*delta
                else:
                    repeated = o.unsqueeze(1).expand_as(f)
                    delta = self.refiner(torch.cat([f,repeated,node_embeddings],dim=-1))
                    selected = (f*region_mask.unsqueeze(-1)).sum(1)/region_mask.sum(1).clamp_min(1).unsqueeze(-1)
                    delta_pool = (delta*region_mask.unsqueeze(-1)).sum(1)/region_mask.sum(1).clamp_min(1).unsqueeze(-1)
                    evidence = self.verify(torch.cat([selected,o,delta_pool],dim=-1))
                    p = evidence.softmax(dim=-1)
                    if self.method == 'full':
                        gate = (p[:,0:1]*torch.sigmoid(
                            self.gate(torch.cat([selected,o,p],dim=-1)))).squeeze(-1)
                    else:
                        gate = torch.ones_like(gate)
                    f = f + active * gate.view(b,1,1)*region_mask.unsqueeze(-1)*delta
            alignment_list.append(attention)
            evidence_list.append(evidence)
            gates.append(gate)
            masks.append(region_mask)
            stages.append(self.heads(f))
        if self.method == 'direct_fusion':
            pooled = (obs*obs_mask.unsqueeze(-1)).sum(1) / obs_mask.sum(1).clamp_min(1).unsqueeze(-1)
            has_obs = (obs_mask.sum(1)>0).float().view(b,1,1)
            f = base + has_obs*self.simple_update(pooled).unsqueeze(1)
            stages = [stages[0]]+[self.heads(f)]*t
        attr,rel,cat = self.heads(f)
        # Retrieval image gallery MUST be independent of the paired query speech.
        # Use the image-only predicted graph, not the progressively fused f.
        pooled_obs = (obs*obs_mask.unsqueeze(-1)).sum(1)/obs_mask.sum(1).clamp_min(1).unsqueeze(-1)
        return {'attributes':attr,'relations':rel,'categories':cat,
                'graph':graph,'features':f,
                'image_embedding':F.normalize(self.image_projection(graph['nodes'].mean(1)),dim=-1),
                'speech_embedding':F.normalize(self.audio_projection(pooled_obs),dim=-1),
                'stage_attributes':torch.stack([x[0] for x in stages],dim=1),
                'stage_relations':torch.stack([x[1] for x in stages],dim=1),
                'stage_categories':torch.stack([x[2] for x in stages],dim=1),
                'alignment':torch.stack(alignment_list,dim=1),
                'evidence':torch.stack(evidence_list,dim=1),
                'gates':torch.stack(gates,dim=1),
                'region_masks':torch.stack(masks,dim=1)}
