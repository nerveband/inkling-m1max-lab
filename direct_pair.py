"""One warm-up and six measured requests per context with a new prompt each time."""
import pathlib
import sys
from direct import run

if __name__=='__main__':
    root=pathlib.Path(sys.argv[1])
    for length,label in ((2048,'2k'),(8192,'8k')):
        run(root/(label+'.jsonl'),length,256,6)
