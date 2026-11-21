import json, os, pathlib, shutil, subprocess, sys, tempfile, threading, time
from proxy import make_proxy
ROOT=pathlib.Path(__file__).resolve().parent
def run(out,held=False):
    out=pathlib.Path(out).resolve(); out.mkdir(parents=True)
    old,new,fn,args,expected=('return a - b','return a + b','add','8, 3',11) if not held else ('return n + 1','return n - 1','previous','19',18)
    final='PATCH VERIFIED' if not held else 'CHECK COMPLETE'
    sig='a, b' if not held else 'n'
    before=f'def {fn}({sig}):\n    {old}\n'
    after=before.replace(old,new)
    with tempfile.TemporaryDirectory(prefix='inkling-task-') as d:
        base=pathlib.Path(d); seed=base/'seed'; fixture=base/'fixture'; agent=base/'agent'
        seed.mkdir();agent.mkdir()
        (seed/'calc.py').write_text(before)
        (seed/'check.py').write_text(f'from calc import {fn}\nassert {fn}({args}) == {expected}\nprint("PASS")\n')
        def git(*args,cwd=seed):
            return subprocess.check_output(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false',*args],cwd=cwd,text=True).strip()
        git('init','-q');git('add','calc.py','check.py');git('commit','-qm','Synthetic input')
        git('worktree','add','-q','--detach',str(fixture),'HEAD')
        (base/'gate.json').write_text(json.dumps(dict(old=old,new=new,before=before,after=after)))
        model_id=os.environ.get('BENCH_MODEL_ID','inkling')
        cfg=json.loads((ROOT/'configs/models.json').read_text())
        cfg['providers']['inkling-local']['models'][0]['id']=model_id
        (agent/'models.json').write_text(json.dumps(cfg))
        env=dict(PATH=os.environ['PATH'],HOME=str(base),TMPDIR=tempfile.gettempdir(),PI_CODING_AGENT_DIR=str(agent),PI_OFFLINE='1',FIXTURE_GATE=str(base/'gate.json'),NO_COLOR='1')
        prompt=f'Read calc.py using read. Use edit to replace the exact substring `{old}` with `{new}` (exclude indentation). Run exactly `python3 -B check.py` using bash. When it passes, respond with exactly {final}. Use relative paths. Perform each step in order.'
        cli=['pi','--offline','--no-session','--no-approve','--no-context-files','--no-extensions','--no-skills','--no-mcp','--no-prompt-templates','--no-themes','-e',str(ROOT/'fixtures/bench.ts'),'--tools','read,edit,bash','--provider','inkling-local','--model',model_id,'--thinking','off','--mode','json','-p',prompt]
        proxy=make_proxy(out); thread=threading.Thread(target=proxy.serve_forever,daemon=True); thread.start()
        try:
            t=time.monotonic()
            with (out/'pi.jsonl').open('w') as f, (out/'stderr').open('w') as e:
                result=subprocess.run(cli,cwd=fixture,env=env,stdout=f,stderr=e,timeout=600)
            elapsed=time.monotonic()-t
        finally: proxy.shutdown();proxy.server_close()
        events=[json.loads(s) for s in (out/'pi.jsonl').read_text().splitlines() if s.startswith('{')]
        calls=[e for e in events if e.get('type')=='tool_execution_start']
        verified=[e for e in events if e.get('type')=='tool_execution_end']
        messages=[e['message'] for e in events if e.get('type')=='message_end' and e['message'].get('role')=='assistant']
        answer=''.join(c.get('text','') for c in messages[-1].get('content',[]) if c.get('type')=='text').strip() if messages else ''
        diff=git('diff','--name-only',cwd=fixture)
        test=subprocess.run(['python3','-B','check.py'],cwd=fixture,capture_output=True)
        required=[]
        for c in calls:
            tool, arguments=c['toolName'],c.get('args',{})
            if tool=='read' and arguments.get('path','').removeprefix('./')=='calc.py' and 'read' not in required:
                required.append('read')
            if tool=='edit' or (tool=='bash' and arguments.get('command')=='python3 -B check.py'):
                required.append(tool)
        checks=dict(exit_ok=result.returncode==0,ordered_tools=required==['read','edit','bash'],
                    tools_ok=len(verified)==len(calls) and all(not v.get('isError') for v in verified),
                    exact_diff=diff=='calc.py' and (fixture/'calc.py').read_text()==after,
                    test_ok=test.returncode==0,answer_ok=answer==final)
        ok=all(checks.values())
        turns=[json.loads(p.read_text()) for p in sorted(out.glob('response-*.json'))]
        summary=dict(passed=ok,checks=checks,wall_s=elapsed,tools=[c['toolName'] for c in calls],answer=answer,test=test.returncode,turns=turns)
        (out/'score.json').write_text(json.dumps(summary,indent=2));print('Pi',ok,round(elapsed,2),summary['tools'],repr(answer[:80]),flush=True)
        return summary
if __name__=='__main__': run(sys.argv[1], '--held' in sys.argv)
