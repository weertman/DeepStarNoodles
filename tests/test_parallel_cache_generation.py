"""Performance-only generator must preserve the frozen serial data contract."""
import csv
import importlib.util
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image
import pytest
import torch

import generate_refreshed_tracer_cache as fast

# Frozen pre-optimization implementation; retained as a compatibility oracle.
FROZEN = Path(__file__).resolve().parent / 'fixtures/serial_cache_generator.py'

@pytest.fixture
def setup(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('frozen_generator_reference', FROZEN)
    serial = importlib.util.module_from_spec(spec); spec.loader.exec_module(serial)
    rows = []
    for i in range(23):
        p = tmp_path / f'image{i}.png'; Image.fromarray(np.full((16,16,3), 20+i*7, np.uint8)).save(p)
        q = tmp_path / f'shapes{i}.json'; q.write_text('[]')
        rows.append({'image_path':str(p),'shapes_path':str(q),'width':'16','height':'16','arm_count':'0'})
    norm=tmp_path/'normalization.json'; norm.write_text(json.dumps({'mean':[0]*18,'std':[1]*18}))
    checkpoint=tmp_path/'parent.pt'; checkpoint.write_bytes(b'parent')
    calls={}; current=[None]
    class Model:
        def __call__(self, batch):
            calls[current[0]].append(batch.clone())
            return batch[:, :1].repeat(1,18,1,1)
    for module in (serial,fast):
        monkeypatch.setattr(module,'REFRESH_IMAGE_SIZE',16)
        monkeypatch.setattr(module,'validate_cache_contract',lambda **kw: None)
        monkeypatch.setattr(module,'resolve_repo_path',lambda p:Path(p))
        monkeypatch.setattr(module,'to_repo_relative',lambda p:str(p))
        monkeypatch.setattr(module,'display_path',lambda p:str(p))
        monkeypatch.setattr(module,'prepared_arms_from_row',lambda row:[])
        augment=module.augment_image_and_arms
        monkeypatch.setattr(module,'augment_image_and_arms',lambda image,arms,params,_f=augment:_f(image,arms,params,size=16))
        monkeypatch.setattr(module,'load_frozen_model',lambda *a:Model())
    def run(module,name,workers=None,draws=1):
        root=tmp_path/name; (root/'indexes').mkdir(parents=True,exist_ok=True)
        (root/'experiment_contract.json').write_text(json.dumps({'cohort':'all','contract_id':'same','frozen_checkpoint_sha256':'same','splits':{'train':{'records':len(rows),'materialized_sha256':'same'}}}))
        with (root/'indexes'/module.index_name('all','train')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        current[0]=name; calls[name]=[]
        kw={} if workers is None else {'cpu_workers':workers}
        path=module.generate(study_root=root,checkpoint=checkpoint,normalization_json=norm,cycle=7,device='cpu',batch_size=4,draws_per_image=draws,**kw)
        return root,path,calls[name]
    return serial,run

@pytest.mark.parametrize('workers',[0,3,10])
def test_same_as_frozen_generator(setup,workers):
    serial,run=setup
    _,a,ac=run(serial,'reference')
    root,b,bc=run(fast,'candidate',workers)
    assert [len(x) for x in ac]==[4,4,4,4,4,3]
    assert len(ac)==len(bc) and all(torch.equal(x,y) for x,y in zip(ac,bc))
    assert (a/'manifest.json').read_bytes()==(b/'manifest.json').read_bytes()
    for p in a.glob('train/*.npz'):
        assert p.read_bytes()==(b/'train'/p.name).read_bytes()
    receipt=json.loads((root/'performance/g0007.json').read_text())
    assert receipt['cpu_workers']==workers and receipt['status']=='complete'
    _,_,reused=run(fast,'candidate',workers)
    assert reused==[]

def test_repeated_draws_and_worker_failure_do_not_publish(setup,monkeypatch):
    serial,run=setup
    _,a,_=run(serial,'reference',draws=2)
    _,b,_=run(fast,'candidate',10,draws=2)
    assert (a/'manifest.json').read_bytes()==(b/'manifest.json').read_bytes()
    def fail(*a,**kw): raise OSError('write failure')
    monkeypatch.setattr(fast,'write_cache_record',fail)
    with pytest.raises(OSError,match='write failure'): run(fast,'failed',10)
    assert not (b.parents[2].parent/'failed/cache/train_manifest.json').exists()

def test_compression_work_is_concurrent_and_bounded(setup,monkeypatch):
    _,run=setup; original=fast.write_cache_record; active=0; peak=0; lock=threading.Lock()
    def slow(*a,**kw):
        nonlocal active,peak
        with lock: active+=1;peak=max(peak,active)
        try: time.sleep(.02); return original(*a,**kw)
        finally:
            with lock: active-=1
    monkeypatch.setattr(fast,'write_cache_record',slow)
    run(fast,'parallel',6)
    assert 1<peak<=4
