from prokg.data import ProKGDataset
from prokg.metrics import recall_at_k,multilabel_macro_f1,multilabel_map
import numpy as np


def test_demo_example():
    data=ProKGDataset('data/demo/train.jsonl','configs/vocab_demo.json',max_observations=3,grid=4)
    item=data[0]
    assert item['image'].shape==(3,128,128)
    assert item['audio'].shape[0]==3
    assert item['node_object_labels'].ge(0).any()
    assert item['evidence'].tolist()==[0,1,2]


def test_recall_uses_unique_image_gallery():
    im=np.array([[1.,0.],[1.,0.],[0.,1.]])
    audio=np.array([[1.,0.],[1.,0.],[0.,1.]])
    assert recall_at_k(im,audio,['a','a','b'],k=1)==100.


def test_metrics_have_independent_scales():
    assert multilabel_macro_f1([[1,0],[0,1]],[[10,-10],[-10,10]])==100.
    assert abs(multilabel_map([[1,0],[0,1]],[[10,-10],[-10,10]])-100.)<0.001
