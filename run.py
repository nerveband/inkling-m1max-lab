"""Supply LLAMA_SERVER and MODEL_DIR, then run: python run.py plain RUN_DIR <job> ..."""
import json, os, pathlib, sys
from guard import guarded
root=pathlib.Path(__file__).resolve().parent
if __name__=='__main__':
    config=json.loads((root/'configs/plain.json').read_text())
    args=config['args'].copy()
    if sys.argv[1]!='plain':
        extra=json.loads((root/'configs'/f'{sys.argv[1]}.json').read_text())
        args+=extra['args']
        config['target']=extra.get('target',config['target'])
        if extra.get('draft'): args+=['--spec-draft-model',str(pathlib.Path(os.environ['MODEL_DIR'])/extra['draft'])]
    command=[os.environ['LLAMA_SERVER'],'-m',str(pathlib.Path(os.environ['MODEL_DIR'])/config['target']),*args]
    guarded(command,sys.argv[3:],sys.argv[2])
