"""Run a server and its benchmark under a single GPU lock on macOS."""
import contextlib, fcntl, json, os, pathlib, plistlib, re, signal, subprocess, sys, tempfile, time

def cmd(*args):
    return subprocess.check_output(args, text=True).strip()

def state():
    pressure=int(cmd('sysctl','-n','kern.memorystatus_vm_pressure_level'))
    swap=float(re.search(r'used = ([\d.]+)M',cmd('sysctl','-n','vm.swapusage'))[1])
    ac="'AC Power'" in cmd('pmset','-g','batt')
    gpu=plistlib.loads(subprocess.check_output(['ioreg','-a','-r','-c','AGXAccelerator','-d','1']))[0]['PerformanceStatistics']
    clients=plistlib.loads(subprocess.check_output(['ioreg','-a','-r','-c','AGXDeviceUserClient','-d','1']))
    inference={}
    for client in clients:
        creator=client.get('IOUserClientCreator','')
        if re.search(r'python|llama|omlx|ollama|\.server|LM Studio|mlx|tensorfold|llmster',creator,re.I):
            pid=int(creator.split(',')[0].split()[-1])
            inference[pid]=inference.get(pid,0)+sum(a.get('accumulatedGPUTime',0) for a in client.get('AppUsage',[]))
    return pressure,swap,ac,gpu,inference

@contextlib.contextmanager
def lock():
    with open(pathlib.Path(tempfile.gettempdir())/'inkling-m1max.gpu.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def reason(s,base,pid):
    pressure,swap,ac,_,clients=s
    if pressure!=1 or swap>base[1] or not ac: return 'pressure, swap growth or AC loss'
    other=[p for p,v in clients.items() if p!=pid and v>base[4].get(p,0)]
    if other and pid:
        # A foreground launcher can own a child inference process. Only
        # processes descending from this exact server PID belong to the job.
        parents={int(row[0]):int(row[1]) for line in cmd('ps','-axo','pid=,ppid=').splitlines() if len(row:=line.split())==2}
        def owned(process):
            while process>1 and process!=pid:process=parents.get(process,0)
            return process==pid
        other=[p for p in other if not owned(p)]
    if other:return 'competing inference GPU activity'

def guarded(server,job,private,port=18089,ready_path='/health',ready_field=None):
    import urllib.request
    private=pathlib.Path(private)
    private.mkdir(exist_ok=True,parents=True)
    with lock(), (private/'server.log').open('w') as log, (private/'monitor.jsonl').open('w') as samples:
        base=state()
        time.sleep(2)
        if why:=reason(state(),base,None): raise RuntimeError(why)
        t=time.monotonic()
        service=subprocess.Popen(server,stdout=log,stderr=log)
        worker=None
        try:
            for _ in range(900):
                if service.poll() is not None: raise RuntimeError(f'server exited {service.returncode}')
                if why:=reason(state(),base,service.pid): raise RuntimeError(why)
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}'+ready_path,timeout=2) as response:
                        if not ready_field:break
                        status=json.load(response)
                        if status.get(ready_field):break
                        if status.get('preload_failures'):raise RuntimeError('model preload failed: '+str(status['preload_failures']))
                    time.sleep(1)
                except OSError: time.sleep(1)
            else: raise TimeoutError('load')
            load_seconds=round(time.monotonic()-t,3)
            (private/'ready.json').write_text(json.dumps(dict(ready_seconds=load_seconds,initial_swap_mib=base[1],initial_gpu_system=base[3].get('In use system memory'))))
            print('ready seconds',load_seconds,flush=True)
            worker=subprocess.Popen(job,start_new_session=True)
            while worker.poll() is None:
                s=state()
                used=s[3]
                rss=cmd('ps','-o','rss=','-p',str(service.pid))
                print(json.dumps(dict(t=round(time.monotonic()-t,3),pressure=s[0],swap_mib=s[1],ac=s[2],gpu_pct=used.get('Device Utilization %'),gpu_system=used.get('In use system memory'),rss_kib=int(rss))),file=samples,flush=True)
                if why:=reason(s,base,service.pid): raise RuntimeError(why)
                time.sleep(1)
            if worker.returncode: raise RuntimeError(f'benchmark exited {worker.returncode}')
        finally:
            if worker and worker.poll() is None:
                os.killpg(worker.pid,signal.SIGTERM)
                worker.wait(timeout=15)
            if service.poll() is None:
                # Disconnect the client, then let llama.cpp finish the current microbatch.
                for _ in range(120):
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/slots',timeout=2) as response:
                            if not any(s.get('is_processing') for s in json.load(response)): break
                    except OSError: time.sleep(1); break
                    time.sleep(1)
                service.terminate()
                service.wait(timeout=180)
