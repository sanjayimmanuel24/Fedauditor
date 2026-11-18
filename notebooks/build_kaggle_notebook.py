"""Builds notebooks/fedauditor_kaggle_qwen.ipynb: python notebooks/build_kaggle_notebook.py (run from Final-Year/)."""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# FedAuditor: open-model audits on a free Kaggle GPU (Qwen3 via Ollama)

Runs the FedAuditor agent with an open model over the FedFault grid. It is the same audit,
tools, privacy budget, fact-checker and scoring as the Claude agent, at zero API cost.

**Setup (once):**
1. Run `python -m fedauditor.make_kaggle_bundle` locally. Upload `results/kaggle/fedauditor_bundle.zip`
   as a **private** Kaggle dataset (e.g. `fedauditor-bundle`).
2. In this notebook: *Add Input* → that dataset. *Settings* → Accelerator **GPU T4 x2**, **Internet on**.
3. First run with `LIMIT = 3` (pilot, about 10 minutes) and check the output. Then set `LIMIT = None` and use
   **Save Version → Save & Run All** to run headless (up to 12 h).

**Resuming:** runs are resumable. To continue after a session ends, add the previous version's
*output* as an input; cell 2 copies its result files back before the run.

**Results:** download `/kaggle/working/fedauditor_results.zip`, unzip into `Final-Year/`, then run
`python -m fedauditor.run_audit_grid --summary-only` locally.
""")

code("""
# 1. Settings
MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"   # Ollama tag: MoE (3B active), NON-thinking variant (~19 GB)
LABEL = "qwen3-30b-instruct"   # results label: audit_grid_brfss_<model>_<LABEL>.jsonl
MAX_TOKENS = 8192       # max reply tokens per turn
MODELS = ["lr"]         # FL models to audit: lr, mlp, xgb (xgb is slow on Kaggle CPUs)
LIMIT = 3               # pilot: clean + poisoning + subgroup_bias (seed 20); None = full grid (125 per model)
WORKERS = 2             # parallel audits (= Ollama parallel slots); 2 leaves KV-cache room for the 30B model
CONTEXT = 40960         # tokens per audit; a full audit needs about 35k
THINKING = False        # Qwen3 thinking mode: better reasoning, much slower
""")

code("""
# 2. Copy the code + data bundle (and any earlier results) into /kaggle/working
import glob, os, shutil
src = glob.glob("/kaggle/input/**/Final-Year/config.py", recursive=True)
assert src, "Attach the fedauditor bundle dataset (Add Input)"
ROOT = "/kaggle/working/Final-Year"
if not os.path.exists(ROOT):
    shutil.copytree(os.path.dirname(src[0]), ROOT)
# Kaggle auto-extracts .gz files in datasets: restore the gzipped BRFSS cache the loader expects
import gzip
gz = f"{ROOT}/data/brfss/brfss_2023.csv.gz"
if not os.path.exists(gz):
    raw = [p for p in glob.glob("/kaggle/input/**/brfss_2023.csv*", recursive=True) if os.path.isfile(p)]
    raw = [p for p in raw if not p.endswith(".gz")] or raw
    assert raw, "brfss_2023.csv not found in the attached dataset"
    os.makedirs(os.path.dirname(gz), exist_ok=True)
    if raw[0].endswith(".gz"):
        shutil.copy(raw[0], gz)
    else:
        with open(raw[0], "rb") as fi, gzip.open(gz, "wb", compresslevel=1) as fo:
            shutil.copyfileobj(fi, fo)
print("BRFSS cache:", gz, round(os.path.getsize(gz) / 1e6, 1), "MB")
res_dir = f"{ROOT}/results/fedauditor"
os.makedirs(f"{res_dir}/audits", exist_ok=True)
# resume: keep the longest copy of each grid file found in any input (e.g. a previous version's output)
for f in glob.glob("/kaggle/input/**/audit_grid_*.jsonl", recursive=True):
    dst = f"{res_dir}/{os.path.basename(f)}"
    if not os.path.exists(dst) or sum(1 for _ in open(f)) > sum(1 for _ in open(dst)):
        shutil.copy(f, dst)
for f in glob.glob("/kaggle/input/**/audits/*.json", recursive=True):
    shutil.copy(f, f"{res_dir}/audits/")
os.chdir(ROOT)
print(os.getcwd(), sorted(os.listdir(res_dir)))
""")

code("""
# 3. Python dependencies for the federation code (numpy/pandas/sklearn come with Kaggle)
!pip install -q "flwr==1.43.0" python-dotenv
import importlib.util
if importlib.util.find_spec("xgboost") is None:
    !pip install -q "xgboost==3.4.1"
import flwr, xgboost, numpy; print("flwr", flwr.__version__, "xgboost", xgboost.__version__, "numpy", numpy.__version__)
""")

code("""
# 4. Install and start Ollama, then pull the model
import shutil, subprocess, time, urllib.request
# the installer needs zstd to unpack (.tar.zst) and lspci to detect the GPUs
!apt-get -qq update > /dev/null && apt-get -qq install -y zstd pciutils > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -n 5
OLLAMA = shutil.which("ollama") or "/usr/local/bin/ollama"
if not os.path.exists(OLLAMA):   # fallback: unpack the release by hand
    !curl -fsSL -o /tmp/ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst
    !tar --use-compress-program=unzstd -xf /tmp/ollama.tar.zst -C /usr/local
    OLLAMA = "/usr/local/bin/ollama"
assert os.path.exists(OLLAMA), "Ollama install failed - see the output above"
env = dict(os.environ, OLLAMA_NUM_PARALLEL=str(WORKERS), OLLAMA_CONTEXT_LENGTH=str(CONTEXT),
           OLLAMA_FLASH_ATTENTION="1", OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_KEEP_ALIVE="-1")
server = subprocess.Popen([OLLAMA, "serve"], env=env, stdout=open("/kaggle/working/ollama.log", "w"),
                          stderr=subprocess.STDOUT)
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:11434"); break
    except Exception:
        time.sleep(1)
else:
    print(open("/kaggle/working/ollama.log").read()[-3000:])
    raise RuntimeError("ollama serve did not start - log above")
for attempt in range(1, 6):   # big pulls sometimes drop with "EOF"; a re-pull resumes
    if subprocess.run([OLLAMA, "pull", MODEL]).returncode == 0:
        break
    print(f"pull attempt {attempt} failed, retrying...")
    time.sleep(10)
!{OLLAMA} list
assert MODEL.split(":")[0] in subprocess.run([OLLAMA, "list"], capture_output=True, text=True).stdout, "model not pulled"
!grep -iE "inference compute|no compatible GPUs" /kaggle/working/ollama.log | tail -n 3
!nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
""")

code("""
# 5. Check that tool calling works through the OpenAI-compatible endpoint
import httpx, json
r = httpx.post("http://localhost:11434/v1/chat/completions", timeout=300, json={
    "model": MODEL, "messages": [{"role": "user", "content": "What is the weather in Paris? /no_think"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
               "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]})
msg = r.json()["choices"][0]["message"]
print(json.dumps(msg.get("tool_calls"), indent=1) if msg.get("tool_calls") else "NO tool_calls: " + str(msg.get("content"))[:300])
""")

code("""
# 6. Run the audits (resumable; prints one line per audit)
# pilot: clean + an easy fault (poisoning) + a hard one (subgroup_bias), low severity, on DEV seed 10
# (not an evaluation seed), saved under a separate "-pilot" label so it never mixes with the grid
limit = (f"--limit {LIMIT} --seeds 10 --severities low --faults poisoning subgroup_bias explanation_instability "
         f"--label {LABEL}-pilot" if LIMIT else f"--label {LABEL}")
think = "--thinking" if THINKING else ""
!python -m fedauditor.run_audit_grid --models {" ".join(MODELS)} --agents openai --llm {MODEL} \\
    --base-url http://localhost:11434/v1 --workers {WORKERS} --max-tokens {MAX_TOKENS} {limit} {think}
""")

code("""
# 7. Package results for download
import zipfile
out = "/kaggle/working/fedauditor_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:   # grid lines, full records and transcripts
    for f in glob.glob(f"{res_dir}/audit_grid_*_{LABEL}*.jsonl") + glob.glob(f"{res_dir}/audits/*__{LABEL}*.json"):
        zf.write(f, os.path.relpath(f, ROOT))
    zf.write(f"{res_dir}/audit_grid_summary_brfss.json", "results/fedauditor/audit_grid_summary_brfss.json")
print(out, round(os.path.getsize(out) / 1e6, 2), "MB")
""")

md("""
### Alternative: vLLM instead of Ollama
vLLM is faster when it runs, but recent releases may not support the T4 (compute capability 7.5).
If you want to try it, replace cells 4–5 with the cell below, then use `--base-url http://localhost:8000/v1`
and `--llm Qwen/Qwen3-8B-AWQ` in cell 6.
""")

code("""
# !pip install -q uv && uv venv /kaggle/working/vllm-env && uv pip install --python /kaggle/working/vllm-env vllm
# vllm_server = subprocess.Popen(["/kaggle/working/vllm-env/bin/vllm", "serve", "Qwen/Qwen3-8B-AWQ",
#     "--dtype", "half", "--tensor-parallel-size", "2", "--max-model-len", "40960",
#     "--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
#     stdout=open("/kaggle/working/vllm.log", "w"), stderr=subprocess.STDOUT)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 5}
OUT = sys.argv[1] if len(sys.argv) > 1 else "notebooks/fedauditor_kaggle_qwen.ipynb"
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", OUT, len(cells), "cells")
