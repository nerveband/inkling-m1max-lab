# Inkling M1 Max lab

A single-user experiment with Inkling 2 Mini on a 24-core M1 Max with 64 GiB of unified memory. The primary measurement comes from a real Pi coding-agent turn. Results and qualification status are in [PLAN.md](PLAN.md). Hardware estimates are in [THEORY.md](THEORY.md).

Set MODEL_DIR to the directory containing verified GGUF files. Set LLAMA_SERVER to the extracted llama-server binary. With Pi 1.6.0 on PATH, run from this checkout using Python 3.12:

```sh
python3 run.py plain measurements/baseline python3 pi_task.py measurements/baseline/pi
python3 run.py plain measurements/direct python3 direct.py measurements/direct/2k.jsonl 2048 256 6
```

The agent selects provider inkling-local and model inkling. A temporary Pi configuration points exclusively at the local measurement proxy on port 18090. The proxy forwards completions to the server on 127.0.0.1:18089. Private job traces and absolute paths stay outside the published measurement set. Use a new measurement directory for each run.
