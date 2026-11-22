"""Start an isolated oMLX with one registered model and run a loopback benchmark."""
import argparse
import json
import os
import pathlib
import sys
import tempfile
from guard import guarded

if __name__=='__main__':
    op=argparse.ArgumentParser()
    op.add_argument('--state',type=pathlib.Path,required=True)
    op.add_argument('--weights',type=pathlib.Path,required=True)
    op.add_argument('--cache',type=pathlib.Path)
    op.add_argument('--context',type=int,default=32768)
    op.add_argument('--depth',type=int,default=0)
    op.add_argument('--slots',type=int)
    op.add_argument('--adaptive',action='store_true')
    op.add_argument('--m1-fast',action='store_true')
    op.add_argument('--affine4',action='store_true')
    op.add_argument('--job',choices=('screen','primary','single','qual'),default='screen')
    op.add_argument('results',type=pathlib.Path)
    a=op.parse_args()
    state=a.state.resolve();results=a.results.resolve()
    state.mkdir(parents=True,exist_ok=False)
    models=state/'models';models.mkdir()
    (models/'inkling').symlink_to(a.weights.resolve(),target_is_directory=True)
    setting={'mtp_enabled':a.depth>0,'max_context_window':a.context}
    if a.depth:
        setting['mtp_adaptive_max_depth' if a.adaptive else 'mtp_fixed_depth']=a.depth
    (state/'model_settings.json').write_text(json.dumps({'version':2,'models':{'inkling':setting}}))
    app=os.environ.get('OMLX_APP')
    if app:
        resources=pathlib.Path(app).resolve()/'Contents/Resources'
        runtime=resources/'Python'
        binary=str(runtime/'cpython-3.11/bin/python3.11')
        modules=os.pathsep.join(map(str,(resources,runtime/'framework-mlx-base/lib/python3.11/site-packages')))
    else:
        resources=None
        binary=os.environ['OMLX_PYTHON']
        modules=os.environ['OMLX_PYTHONPATH']
    with tempfile.TemporaryDirectory(prefix='inkling-omlx-overlay-') as overlay_tmp:
        if a.m1_fast or a.affine4:
            if resources is None:
                p_err='OMLX_APP must be set when using --m1-fast or --affine4.'
                raise SystemExit(p_err)
            from m1_overlay import M1_FAST_FEATURES, PATCHES, build_overlay
            raw_feats=os.environ.get('OMLX_OVERLAY_FEATURES')
            if raw_feats:
                feats={x.strip() for x in raw_feats.split(',') if x.strip()}
            else:
                feats=set(M1_FAST_FEATURES) if a.m1_fast else set()
            if a.affine4:feats.add('affine4')
            build_overlay(pathlib.Path(overlay_tmp),feats,resources)
            modules=os.pathsep.join((overlay_tmp,str(PATCHES),modules))
        command=['env','PYTHONPATH='+modules,'HOME='+str(state),'OMLX_BASE_PATH='+str(state),binary,'-m','omlx.cli','serve','--base-path',str(state),'--model-dir',str(models),'--host','127.0.0.1','--port','18089','--headless']
        if a.slots is not None:
            command+=['--max-concurrent-requests',str(a.slots)]
        if a.cache:
            cache=a.cache.resolve();cache.mkdir(parents=True,exist_ok=False)
            command+=['--paged-ssd-cache-dir',str(cache),'--paged-ssd-cache-max-size','2GB','--hot-cache-max-size','1GB']
        else:command+=['--no-cache']
        root=pathlib.Path(__file__).resolve().parent
        os.environ.update(HF_HUB_OFFLINE='1',BENCH_MODEL_ID='inkling',BENCH_TOKENIZER='omlx')
        script={'screen':'mlx_screen.py','primary':'primary.py','single':'warm_pi.py','qual':'qualify_llama.py'}[a.job]
        job=[sys.executable,'-u',str(root/script),str(results)]
        if a.job=='qual':job.append(str(a.context))
        guarded(command,job,results,ready_path='/v1/models')
