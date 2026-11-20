import torch
import pytest
from prokg.model import ProKGVision,METHODS
from prokg.losses import prokg_loss

@pytest.mark.parametrize('method',METHODS)
def test_all_model_variants_forward(method):
    torch.manual_seed(3)
    model=ProKGVision(4,4,4,3,method=method,dim=32,grid=4)
    out=model(torch.rand(2,3,64,64),torch.zeros(2,2,8000),
              torch.zeros(2,2,32,dtype=torch.long),torch.tensor([[1.,1.],[1.,0.]]))
    assert out['attributes'].shape==(2,4)
    assert out['graph']['edges'].shape==(2,16,16,4)
    assert out['evidence'].shape==(2,2,3)
    assert out['stage_attributes'].shape==(2,3,4)
    assert torch.isfinite(out['features']).all()


def test_no_observations_do_not_change_features():
    model=ProKGVision(4,4,4,3,method='full',dim=32,grid=4).eval()
    with torch.no_grad():
        output=model(torch.rand(1,3,64,64),torch.zeros(1,2,8000),
                     torch.zeros(1,2,32,dtype=torch.long),torch.zeros(1,2))
    assert torch.allclose(output['stage_attributes'][:,0],output['stage_attributes'][:,-1],atol=1e-6)


def test_verification_loss_can_ignore_missing_evidence():
    model=ProKGVision(4,4,4,3,method='full',dim=32,grid=4)
    out=model(torch.rand(2,3,64,64),torch.zeros(2,2,8000),
              torch.ones(2,2,32,dtype=torch.long),torch.ones(2,2))
    batch={'attributes':torch.zeros(2,4),'relations':torch.zeros(2,4),
           'categories':torch.zeros(2,3),'has_attr':torch.zeros(2),
           'has_rel':torch.zeros(2),'has_cat':torch.zeros(2),
           'has_retrieval':torch.zeros(2),'evidence':torch.full((2,2),-100),
           'obs_mask':torch.ones(2,2),'target_center':torch.full((2,2,2),-1.),
           'image_id':['x','y']}
    loss,components=prokg_loss(out,batch)
    assert float(components['verification'].detach())==0.
    assert torch.isfinite(loss)


def test_retrieval_image_embedding_does_not_use_matching_speech():
    model=ProKGVision(4,4,4,3,method='full',dim=32,grid=4).eval()
    image=torch.rand(1,3,64,64)
    audio=torch.rand(1,2,8000)
    with torch.no_grad():
        a=model(image,audio,torch.ones(1,2,32,dtype=torch.long),torch.ones(1,2))
        b=model(image,audio,torch.zeros(1,2,32,dtype=torch.long),torch.ones(1,2))
    assert torch.allclose(a['image_embedding'],b['image_embedding'],atol=1e-6)
