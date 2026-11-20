import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

EVIDENCE = {"supported": 0, "contradicted": 1, "insufficient": 2}


def tokenize(text, max_tokens=32, bins=8192):
    """Stable hashed word IDs; replace with a pretrained tokenizer in a full-scale run."""
    out = []
    for word in text.lower().split()[:max_tokens]:
        digest = hashlib.sha1(word.encode('utf8')).digest()
        out.append(1 + int.from_bytes(digest[:4], 'little') % (bins - 1))
    return out + [0] * (max_tokens - len(out))


def read_manifest(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_vocab(path):
    with open(path, encoding='utf8') as f:
        vocabulary = json.load(f)
    for name in ('objects', 'attributes', 'relations', 'categories'):
        if name not in vocabulary or not isinstance(vocabulary[name], list):
            raise ValueError(f'Vocabulary must contain a list for {name}')
    return vocabulary


def image_tensor(path, image_size):
    with Image.open(path) as image:
        image = image.convert('RGB').resize((image_size, image_size))
        raw = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(raw)


def audio_tensor(path, sample_rate, max_audio_seconds):
    total = int(sample_rate * max_audio_seconds)
    if not path:
        return torch.zeros(total, dtype=torch.float32)
    wave, sr = sf.read(path, dtype='float32')
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    wave = torch.tensor(wave, dtype=torch.float32)
    if sr != sample_rate and wave.numel():
        # torch interpolation avoids requiring external speech dependencies.
        new_length = max(1, int(round(wave.numel() * sample_rate / sr)))
        wave = F.interpolate(wave[None, None], size=new_length, mode='linear',
                             align_corners=False).flatten()
    if wave.numel() < total:
        wave = F.pad(wave, (0, total - wave.numel()))
    return wave[:total]


class ProKGDataset(Dataset):
    """Manifest paths may be absolute or relative to repository root."""
    def __init__(self, manifest, vocab, image_size=128, max_observations=4,
                 sample_rate=16000, max_audio_seconds=2.0, grid=4, repo_root=None):
        self.items = read_manifest(manifest)
        self.vocab = load_vocab(vocab) if isinstance(vocab, (str, Path)) else vocab
        self.root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
        self.size = image_size
        self.max_obs = max_observations
        self.sample_rate = sample_rate
        self.max_sec = max_audio_seconds
        self.grid = grid
        self.obj_idx = {s:i for i,s in enumerate(self.vocab['objects'])}
        self.attr_idx = {s: i for i,s in enumerate(self.vocab['attributes'])}
        self.rel_idx = {s: i for i,s in enumerate(self.vocab['relations'])}
        self.cat_idx = {s: i for i,s in enumerate(self.vocab['categories'])}

    def __len__(self):
        return len(self.items)

    def path(self, p):
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    def __getitem__(self, ix):
        record = self.items[ix]
        image = image_tensor(self.path(record['image']), self.size)
        attrs = torch.zeros(len(self.attr_idx))
        rels = torch.zeros(len(self.rel_idx))
        cats = torch.zeros(len(self.cat_idx))
        for label in record.get('labels', {}).get('attributes', []):
            if label in self.attr_idx: attrs[self.attr_idx[label]] = 1
        for label in record.get('labels', {}).get('relations', []):
            if label in self.rel_idx: rels[self.rel_idx[label]] = 1
        for label in record.get('labels', {}).get('categories', []):
            if label in self.cat_idx: cats[self.cat_idx[label]] = 1
        observations = record.get('observations', [])[:self.max_obs]
        audios = torch.stack([audio_tensor(str(self.path(o['audio'])) if o.get('audio') else '',
                                self.sample_rate, self.max_sec) for o in observations] +
                             [torch.zeros(int(self.sample_rate * self.max_sec))] * (self.max_obs - len(observations)))
        texts = torch.tensor([tokenize(o.get('text', '')) for o in observations] +
                             [[0]*32] * (self.max_obs - len(observations)), dtype=torch.long)
        obs_mask = torch.tensor([1.] * len(observations) + [0.] * (self.max_obs-len(observations)))
        evidence = torch.tensor([EVIDENCE.get(o.get('evidence'), -100) for o in observations] +
                                [-100] * (self.max_obs-len(observations)), dtype=torch.long)
        # Supervision on spatial grounding is obtained from a region centre, not
        # from ground-truth graph nodes at inference.
        centers = []
        for o in observations:
            box = o.get('target_box')
            centers.append([(float(box[0])+float(box[2]))/2,
                            (float(box[1])+float(box[3]))/2] if box else [-1., -1.])
        centers += [[-1.,-1.]] * (self.max_obs - len(observations))
        # Graph supervision is populated only for training losses. Missing
        # object/edge annotation remains -100 (unknown, NOT a negative label).
        n = self.grid * self.grid
        node_objects = torch.full((n,),-100,dtype=torch.long)
        node_attributes = torch.full((n,len(self.attr_idx)),-1.,dtype=torch.float32)
        edge_relations = torch.full((n,n),-100,dtype=torch.long)
        assigned = {}
        for obj in record.get('objects',[]):
            box = obj.get('bbox')
            if not box or obj.get('category') not in self.obj_idx: continue
            col=min(self.grid-1,max(0,int((float(box[0])+float(box[2]))*0.5*self.grid)))
            row=min(self.grid-1,max(0,int((float(box[1])+float(box[3]))*0.5*self.grid)))
            spot = row*self.grid+col
            assigned[str(obj.get('id'))] = spot
            node_objects[spot] = self.obj_idx[obj['category']]
            object_attrs=[a for a in obj.get('attributes',[]) if a in self.attr_idx]
            if object_attrs:
                node_attributes[spot] = 0.
                for a in object_attrs: node_attributes[spot,self.attr_idx[a]] = 1.
        for relation in record.get('relations',[]):
            sid=assigned.get(str(relation.get('subject_id')))
            oid=assigned.get(str(relation.get('object_id')))
            predicate=relation.get('predicate')
            if sid is not None and oid is not None and predicate in self.rel_idx:
                edge_relations[sid,oid]=self.rel_idx[predicate]
        return {'image': image, 'audio': audios, 'tokens': texts,
                'node_object_labels':node_objects,
                'node_attribute_labels':node_attributes,
                'edge_relation_labels':edge_relations,
                'obs_mask': obs_mask, 'evidence': evidence,
                'target_center': torch.tensor(centers, dtype=torch.float32),
                'attributes': attrs, 'relations': rels, 'categories': cats,
                'has_attr': torch.tensor(float('attributes' in record.get('labels', {}))),
                'has_rel': torch.tensor(float('relations' in record.get('labels', {}))),
                'has_cat': torch.tensor(float('categories' in record.get('labels', {}))),
                'has_retrieval': torch.tensor(float(bool(observations))),
                'image_id': str(record.get('image_id', record.get('sample_id', ix))),
                'sample_id': str(record.get('sample_id', ix)),
                'source': str(record.get('source','unknown'))}


def loader(manifest, vocab, batch_size=8, shuffle=False, workers=0, **kwargs):
    dataset = ProKGDataset(manifest, vocab, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers)
