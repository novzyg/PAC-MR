"""Synthetic model checks; run with python tests/test_model.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_model():
    import numpy as np
    import torch
    from models import ConflictModel as Revision

    torch.manual_seed(7)
    adj=np.zeros((6,6),dtype=np.float32)
    adj[0,1]=adj[1,0]=adj[1,2]=adj[2,1]=1
    m=Revision([8,5,6],adj,dim=8,dropout=.2,variant='full')
    b={'diag':torch.tensor([[[1,2]],[[2,3]]]),'proc':torch.tensor([[[1]],[[2]]]),'length':torch.ones(2,dtype=torch.long)}
    m.alpha.data.fill_(-10.) # avoid displacement cap in synthetic gradient test
    m.train()
    assert not m.backbone.training
    z,raw,s=m.forward_details(b)
    assert torch.isfinite(z).all() and (z<=raw+1e-6).all()
    assert torch.equal(z[:,3:],raw[:,3:])
    zb,rb,_=m.forward_details(b,mode='bypass'); assert torch.equal(zb,rb)
    ze,_,_=m.forward_details(b,mode='equal'); torch.testing.assert_close(z,ze) # same starting function
    (z * torch.arange(1,7)).sum().backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert m.alpha.grad is not None and torch.isfinite(m.alpha.grad) and m.alpha.grad.abs()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.cost.parameters())
    # Analytic direction agrees with autograd through frozen nonlinear head.
    h,u=m.backbone(b);u=u.detach().requires_grad_()
    g=torch.autograd.grad(m.backbone.predict(h,u).sum(),u)[0]
    f,_,last=m.backbone.head
    expected=(((f(torch.cat([h[:,None].expand_as(u),u],-1))>0)*last.weight[0]) @ f.weight[:,8:])
    torch.testing.assert_close(g,expected)
    print('PASS: descent, bypass identity, isolates, equal initialization, trainable gradients, frozen backbone, analytic direction')

    # Public interface follows SafeDrug's two-value return in train and eval modes.
    m.eval()
    visits=[[[0,1],[0],[1,2]],[[2],[1],[3]]]
    z,loss=m(visits)
    assert z.shape==(1,6) and loss.ndim==0
    other=[[[0,1],[0],[5]],[[2],[1],[]]]
    z2,loss2=m(other)
    torch.testing.assert_close(z,z2)
    torch.testing.assert_close(loss,loss2)
    print('PASS: SafeDrug prefix input / two outputs / no medication-label input leakage')

    # Padding and missing-code visits cannot create NaNs or change other patients.
    from data import collate, training_cooccurrence
    rows = [([([0,1],[]),([],[])],torch.zeros(6),0,1),
            ([(list(range(8)),list(range(5)))],torch.zeros(6),1,0)]
    m.eval()
    one=collate(rows[:1]); together=collate(rows)
    torch.testing.assert_close(m(one)[0],m(together)[0][:1],atol=1e-6,rtol=1e-5)
    assert torch.isfinite(m(together)[0]).all()
    records=[[[[0],[0],[0,1]]], [[[1],[1],[2,3]]]]
    graph=training_cooccurrence(records,[0],6)
    assert graph[0,1]==1 and graph[2,3]==0 and np.diag(graph).sum()==0
    base=Revision([8,5,6],adj,dim=8,dropout=0,variant='base',cooccurrence=graph)
    base(together)[0].square().mean().backward()
    for module in [base.backbone.diag_score, base.backbone.history_query,
                   base.backbone.ehr_encoder,base.backbone.ddi_encoder,
                   base.backbone.cross_attention]:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in module.parameters()), type(module)
    import io
    buffer=io.BytesIO();torch.save(base.state_dict(),buffer);buffer.seek(0)
    copy=Revision([8,5,6],adj,dim=8,dropout=0,variant='base')
    copy.load_state_dict(torch.load(buffer,weights_only=True));base.eval();copy.eval()
    torch.testing.assert_close(base(together)[0],copy(together)[0],atol=0,rtol=0)
    print('PASS: padding invariance, empty codes, train-only graph, new-module gradients, checkpoint reload')

    # Hyperparameter boundaries, default compatibility, and equal allocation at R=1.
    reference=Revision([8,5,6],adj,dim=8,dropout=0)
    explicit=Revision([8,5,6],adj,dim=8,dropout=0,displacement_cap=.1,cost_ratio=4)
    explicit.load_state_dict(reference.state_dict(),strict=True)
    reference.eval();explicit.eval()
    torch.testing.assert_close(reference(b)[0],explicit(b)[0],rtol=0,atol=0)
    equal=Revision([8,5,6],adj,dim=8,dropout=0,cost_ratio=1,displacement_cap=.03)
    equal.load_state_dict(reference.state_dict(),strict=True)
    with torch.no_grad():
        equal.cost[-1].weight.normal_()
    equal.eval()
    z,_,stats=equal.forward_details(b,diagnostics=True)
    ze,_,_=equal.forward_details(b,mode='equal')
    torch.testing.assert_close(z,ze)
    assert stats['allocation_asymmetry']==0
    assert equal.displacement_cap==.03
    for kwargs in [{'displacement_cap':0},{'displacement_cap':float('nan')},{'cost_ratio':.5},{'cost_ratio':float('inf')}]:
        try:
            Revision([8,5,6],adj,dim=8,**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    print('PASS: hyperparameter defaults, strict weight compatibility, R=1 equal allocation, invalid bounds')


if __name__ == "__main__":
    test_model()
