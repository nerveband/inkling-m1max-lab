"""Pi control leg. One warm-up followed by six independent read/edit/test tasks."""
import json,pathlib,sys
from pi_task import run
from summary import summarize
if __name__=='__main__':
    out=pathlib.Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True)
    for i in range(7):
        run(out/f'pi-{i}')
    rows=summarize(out)
    (out/'leg-summary.json').write_text(json.dumps(rows,indent=2))
