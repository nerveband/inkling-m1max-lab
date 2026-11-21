"""Reopen one disposable Pi conversation through its local provider for six checkpoints."""
import json
import os
import pathlib
import subprocess
import tempfile
import threading
import time
import uuid
from proxy import make_proxy

ROOT=pathlib.Path(__file__).resolve().parent

def soak(destination):
    destination=pathlib.Path(destination).resolve()
    destination.mkdir(parents=True)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='inkling-soak-') as d:
        base=pathlib.Path(d);agent=base/'agent';agent.mkdir()
        (agent/'models.json').write_bytes((ROOT/'configs/models.json').read_bytes())
        (base/'gate.json').write_text('{}')
        env=dict(PATH=os.environ['PATH'],HOME=d,PI_CODING_AGENT_DIR=str(agent),PI_OFFLINE='1',FIXTURE_GATE=str(base/'gate.json'),NO_COLOR='1')
        session=str(uuid.uuid4())
        common=['pi','--offline','--session-id',session,'--session-dir',str(base/'sessions'),'--no-context-files','--no-extensions','--no-skills','--no-mcp','--no-prompt-templates','--no-themes','--no-tools','-e',str(ROOT/'fixtures/bench.ts'),'--provider','inkling-local','--model','inkling','--thinking','off','--mode','json','-p']
        proxy=make_proxy(destination)
        thread=threading.Thread(target=proxy.serve_forever,daemon=True);thread.start()
        try:
            for index in range(6):
                prompt='Remember the code CEDAR83. Respond with only the code.' if index==0 else f'Checkpoint {index}. Respond with only the code from my first message.'
                t=time.monotonic()
                p=subprocess.run([*common,prompt],cwd=base,env=env,capture_output=True,text=True,timeout=300)
                messages=[event['message'] for line in p.stdout.splitlines() if line.startswith('{') and (event:=json.loads(line)).get('type')=='message_end' and event['message'].get('role')=='assistant']
                answer=''.join(part['text'] for part in messages[-1]['content'] if part['type']=='text').strip() if messages else ''
                path=destination/f'request-{index}.json'
                history=sum(m['role']=='user' for m in json.loads(path.read_text())['messages']) if path.exists() else 0
                rows.append({'turn':index+1,'history_user_messages':history,'answer':answer,'wall_s':time.monotonic()-t,'passed':p.returncode==0 and answer=='CEDAR83' and history==index+1})
                (destination.parent/'pi-session.json').write_text(json.dumps(rows,indent=2)+'\n')
                if p.returncode:
                    (destination/'stderr').write_text(p.stderr)
                    break
        finally:proxy.shutdown();proxy.server_close()
    return rows
