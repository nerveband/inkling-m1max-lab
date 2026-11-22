"""Serve the pinned oMLX recipe in the foreground with a separate state directory."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import plistlib
import signal
import subprocess
import tempfile

DIGEST='6316ad32cde3248bcf23abece8b9222847f7ae8e594cea1b378a34049252df78'

@contextlib.contextmanager
def application(app,dmg):
    if app:
        yield app.resolve()
        return
    if not dmg or not dmg.is_file():
        raise SystemExit('Supply --app or --dmg pointing at the pinned oMLX 0.8.2 release.')
    with dmg.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=DIGEST:
            raise SystemExit('oMLX disk image SHA-256 mismatch')
    with tempfile.TemporaryDirectory(prefix='inkling-omlx-') as folder:
        mount=Path(folder)/'release';mount.mkdir()
        subprocess.run(['hdiutil','attach','-readonly','-nobrowse','-mountpoint',str(mount),str(dmg.resolve())],check=True)
        try:yield mount/'oMLX.app'
        finally:subprocess.run(['hdiutil','detach',str(mount)],check=True)

def sorted_expert_package(resources, folder):
    """Shadow only the pinned model module for the small-batch expert-sort experiment."""
    source=resources
    dest=folder
    for component in ('omlx','patches','inkling2','model.py'):
        for entry in source.iterdir():
            if entry.name not in (component,'__pycache__'):
                (dest/entry.name).symlink_to(entry,target_is_directory=entry.is_dir())
        source=source/component
        dest=dest/component
        if source.is_dir():dest.mkdir()
    original=source.read_text()
    threshold='do_sort = inds.size >= 64'
    if original.count(threshold)!=1:raise RuntimeError('Unexpected release model source')
    dest.write_text(original.replace(threshold,'do_sort = inds.size >= 32'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--app',type=Path)
    source.add_argument('--dmg',type=Path)
    parser.add_argument('--weights',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--context',type=int,default=32768)
    parser.add_argument('--depth',type=int,default=6)
    parser.add_argument('--slots',type=int,choices=(1,2),default=1)
    parser.add_argument('--sort-small-expert-batches',action='store_true',help='Experimentally sort expert work at 32 indices instead of 64.')
    parser.add_argument('--m1-fast',action='store_true',help='Enable verified M1 Max fused decode, verify attention, and prefill optimizations.')
    parser.add_argument('--affine4',action='store_true',help='Convert NVFP4 experts to affine 4-bit (group size 64) at load time.')
    parser.add_argument('--hot-cache-max-size',default='8GB',help='Hot prefix KV cache size in RAM (default: 8GB).')
    parser.add_argument('--default-temp',type=float,default=0.0,help='Default sampling temperature written to state/settings.json (default: 0.0).')
    parser.add_argument('--features',default=None,help='Optional comma-separated overlay feature list overriding M1_FAST_FEATURES.')
    args=parser.parse_args()
    weights,state,cache=(p.expanduser().resolve() for p in (args.weights,args.state,args.cache))
    if not (weights/'mtp.safetensors').is_file():
        raise SystemExit('The native oQ4e target directory must include mtp.safetensors.')
    marker=state/'inkling-lab-state'
    if state.exists() and not marker.is_file():
        raise SystemExit('Choose a new state directory or reuse one created by this launcher.')
    state.mkdir(parents=True,exist_ok=True);marker.touch()
    models=state/'models';models.mkdir(exist_ok=True)
    link=models/'inkling'
    if not link.is_symlink():link.symlink_to(weights,target_is_directory=True)
    if link.resolve()!=weights:raise SystemExit('State belongs to another weights directory.')
    cache.mkdir(parents=True,exist_ok=True)
    config={'mtp_enabled':args.depth>0,'max_context_window':args.context}
    if args.depth:config['mtp_fixed_depth']=args.depth
    (state/'model_settings.json').write_text(json.dumps({'version':2,'models':{'inkling':config}},indent=2)+'\n')
    settings_path=state/'settings.json'
    settings_data={}
    if settings_path.is_file():
        try:settings_data=json.loads(settings_path.read_text())
        except Exception:settings_data={}
    settings_data.setdefault('sampling',{})['temperature']=float(args.default_temp)
    settings_path.write_text(json.dumps(settings_data,indent=2)+'\n')
    with application(args.app,args.dmg) as app, tempfile.TemporaryDirectory(prefix='inkling-kernel-') as overlay:
        contents=app/'Contents'
        with (contents/'Info.plist').open('rb') as f:
            if plistlib.load(f)['CFBundleShortVersionString']!='0.8.2':raise SystemExit('Expected oMLX 0.8.2.')
        resources=contents/'Resources';py=resources/'Python'
        modules=os.pathsep.join(map(str,(resources,py/'framework-mlx-base/lib/python3.11/site-packages')))
        feats=set()
        if args.features or args.m1_fast or args.affine4:
            from m1_overlay import M1_FAST_FEATURES, PATCHES, build_overlay
            if args.features:
                feats={x.strip() for x in args.features.split(',') if x.strip()}
            elif args.m1_fast:
                feats=set(M1_FAST_FEATURES)
            if args.affine4:feats.add('affine4')
            build_overlay(Path(overlay),feats,resources)
            modules=os.pathsep.join((overlay,str(PATCHES),modules))
        elif args.sort_small_expert_batches:
            sorted_expert_package(resources,Path(overlay))
            modules=overlay+os.pathsep+modules
        environment=dict(os.environ,PYTHONPATH=modules,HF_HUB_OFFLINE='1',HF_HOME=str(cache/'hub'),PYTHONDONTWRITEBYTECODE='1')
        if 'suffix_policy' in feats:
            environment['OMLX_MTP_DRAFT_POLICY']='1'
        command=[str(py/'cpython-3.11/bin/python3.11'),'-m','omlx.cli','serve','--base-path',str(state),'--model-dir',str(models),'--host','127.0.0.1','--port','18089','--max-concurrent-requests',str(args.slots),'--headless','--paged-ssd-cache-dir',str(cache/'kv'),'--paged-ssd-cache-max-size','2GB','--hot-cache-max-size',str(args.hot_cache_max_size)]
        with subprocess.Popen(command,env=environment,start_new_session=True) as server:
            def stop(*_):
                if server.poll() is None:server.terminate()
            previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
            try:return server.wait()
            finally:
                for s,handler in previous.items():signal.signal(s,handler)
                stop()

if __name__=='__main__':raise SystemExit(main())
