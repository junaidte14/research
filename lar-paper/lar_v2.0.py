# %% [markdown]
# # LAR v9.3 — controlled re-run, resumable Kaggle notebook
# 
# Same protocol and config fingerprint as v9 / v9.1, so **existing results files stay compatible** (rows from earlier versions simply have empty new-metric columns).
# 
# **What is new in v9.3:** cell 6 `dataset_statistics()` computes, for every dataset/scale used, the size, label balance, length (words, word-pieces, % truncated), perturbation-candidate coverage and bank acceptance numbers for the paper's Datasets section (writes `dataset_stats.json` and `tab_datasets.tex`). Nothing else changed, so v9.2 results and phases remain valid.
# 
# **What changed since v9.1**
# * **λ locked to 0.25** for the new datasets (SST-2 large sweep, seeds 0–4: λ = 0.25 gives ρ ≈ 0.953 vs 0.958 at λ = 0.5 with no collapsed seed; λ = 0.1 gives 0.933). Phases F and G therefore use `lar_pearson_l0.25`, decided *before* seeing their data.
# * **Phase E2 now runs only the two baselines** (`ce_only_control`, `aug_only`) with the new metrics, because the λ variants already have them.
# * **Pooled new-metric analysis** (`newmetrics_vs_aug` in `run_analysis()`): pairs rows that have top-k overlap / IG stability across groups by seed, e.g. H λ rows against E2 baselines.
# * **`tradeoff_report()` now uses common seeds by default**, so means are paired and comparable (earlier tables compared 10-seed baselines with 5-seed λ rows).
# 
# **Metrics per trained model:** full-validation accuracy; stability (naive-positional vs greedy vs order-preserving alignment); alignment coverage; prediction-flip rate; faithfulness (comprehensiveness / sufficiency vs random ranking); `topk_jaccard` (top-20% overlap, *not optimised by any loss*), `rho_pearson` (matches the Pearson loss, reported for transparency), `ig_rho` (Integrated-Gradients stability, medium strength, 300 pairs).
# 
# **Every session:** run cells 1–5 (definitions), then the phase cell you want. Finished `(phase, condition, seed)` triples are skipped; a half-trained model resumes from its checkpoint; a changed config raises an error. The runner is seed-outer: after each seed every condition has paired data, so you can stop at any point and analyse.
# 
# **Setup / parallel notebooks:** Internet ON, GPU on, Secrets `KAGGLE_USERNAME` / `KAGGLE_KEY`; attach the dataset holding your results store as an input and check that cell 1 prints the expected `Results store: N rows`. Keep `DATASETS`, `SCALES`, `BANK_PARAMS`, `LEARNING_RATE` identical across notebooks. If two notebooks run at once, give each its own `KAGGLE_DATASET_SLUG` (later commits overwrite earlier ones otherwise) and merge result files at analysis time.

# %%
# ===== CELL 1: setup, configuration, persistence (run first, every session) =====
import contextlib
import difflib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

PROTOCOL_VERSION = "v9.0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")

# ---------------------------------------------------------------- paths
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else os.path.abspath("./work_v9")
RUN_DIR = os.path.join(WORK, "lar_v9")
CKPT_DIR = os.path.join(RUN_DIR, "ckpt")      # flat files: ckpt__<phase>__<cond>__seed<k>.pt
BANK_DIR = os.path.join(RUN_DIR, "banks")     # flat files: bank__*.json
STAGE_DIR = os.path.join(WORK, "lar_v9_stage")  # what gets uploaded (flat, no subfolders)
RESULTS_JSON = os.path.join(RUN_DIR, "lar_v9_results.json")
RESULTS_CSV = os.path.join(RUN_DIR, "lar_v9_results.csv")
for d in (RUN_DIR, CKPT_DIR, BANK_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- Kaggle persistence config  (EDIT)
KAGGLE_DATASET_SLUG = "lar-v9-results"       # NEW slug on purpose: never mix with the v8 dataset
KAGGLE_DATASET_TITLE = "LAR v9 Results"
AUTO_COMMIT_TO_KAGGLE = True                 # best-effort; never blocks training

# ---------------------------------------------------------------- experiment config
SEEDS = list(range(10))                       # seeds 0-9 for EVERY phase
LEARNING_RATE = 3e-5
TRAIN_STRENGTH = 0.35                         # perturbation strength used for training
EVAL_STRENGTHS = {"light": 0.15, "medium": 0.35, "heavy": 0.60}
BANK_PARAMS = dict(sim_threshold=0.85,        # sentence-embedding cosine floor (fixed in advance)
                   max_draws_eval=10,         # drafts tried per sentence/strength at eval
                   train_variants=2,          # validated variants kept per training sentence
                   max_draws_train=4,
                   max_synsets=3)             # only the 3 most frequent WordNet senses per word
REF_CLASSIFIER = "distilbert-base-uncased-finetuned-sst-2-english"   # independent label-preservation check
REF_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"             # independent semantic-similarity check
FAITH_FRACS = [0.1, 0.2, 0.3, 0.5]
FAITH_N = 400                                 # sentences used for faithfulness (fixed random subset)
IG_STEPS = 16                                 # Integrated-Gradients steps for the second attribution method
IG_N = 300                                    # medium-strength pairs used for the IG-stability metric
TOPK_FRAC = 0.2                               # top-k overlap uses the top 20% of aligned tokens
LOG_EVERY = 250
BASE_MODEL_NAME = "distilbert-base-uncased"

DATASETS = {
    "sst2": dict(hf=("nyu-mll/glue", "sst2"), text="sentence", train="train", eval="validation",
                 max_len=64, truncate_words=None, shuffle_train=False, eval_n=None),
    "rotten_tomatoes": dict(hf=("rotten_tomatoes", None), text="text", train="train", eval="validation",
                            max_len=64, truncate_words=None, shuffle_train=True, eval_n=None),
    "imdb": dict(hf=("imdb", None), text="text", train="train", eval="test",
                 max_len=128, truncate_words=90, shuffle_train=True, eval_n=872),
}
SCALES = {
    "smoke": dict(train_size=120, epochs=1, ckpt_every=60, commit_ckpt_every=None, eval_limit=60),
    "small": dict(train_size=1500, epochs=1, ckpt_every=500, commit_ckpt_every=None, eval_limit=None),
    "medium": dict(train_size=5000, epochs=1, ckpt_every=1000, commit_ckpt_every=2500, eval_limit=None),
    "large": dict(train_size=20000, epochs=2, ckpt_every=2000, commit_ckpt_every=6000, eval_limit=None),
}
CONDITIONS = {
    "ce_only_control":  dict(kind="ce"),
    "aug_only":         dict(kind="aug"),                                   # 0.5*(CE(x)+CE(x~))
    "lar_pos":          dict(kind="lar", align="positional", dist="cos", lam=0.5),   # LAR exactly as in v8
    "lar_aligned":      dict(kind="lar", align="seq", dist="cos", lam=0.5),          # alignment-aware training loss
    "lar_pearson":      dict(kind="lar", align="seq", dist="pearson", lam=0.5),      # FAR-style correlation loss
    "pred_consistency": dict(kind="kl", lam=0.5),                           # symmetric KL between predictions
    "lar_pearson_l0.1":  dict(kind="lar", align="seq", dist="pearson", lam=0.1),    # lambda sweep (accuracy/stability trade-off)
    "lar_pearson_l0.25": dict(kind="lar", align="seq", dist="pearson", lam=0.25),
}
PHASES = {
    "S_smoke":               dict(dataset="sst2", scale="smoke", conditions=list(CONDITIONS)),
    "A_sst2_small_core":     dict(dataset="sst2", scale="small",
                                  conditions=["ce_only_control", "aug_only", "lar_pos"]),
    "B_sst2_small_variants": dict(dataset="sst2", scale="small",
                                  conditions=["lar_aligned", "lar_pearson", "pred_consistency"]),
    "C_rt_small":            dict(dataset="rotten_tomatoes", scale="small",
                                  conditions=["ce_only_control", "aug_only", "lar_pos", "lar_aligned"]),
    "D_imdb_small":          dict(dataset="imdb", scale="small",
                                  conditions=["ce_only_control", "aug_only", "lar_pos", "lar_aligned"]),
    "E_sst2_large":          dict(dataset="sst2", scale="large",
                                  conditions=["ce_only_control", "aug_only", "lar_pos", "lar_aligned", "lar_pearson"]),
    # ---- v9.1/v9.2 phases (all add new metrics: top-k overlap, aligned Pearson, IG stability)
    # H: lambda sweep for the Pearson loss at SST-2 large. 0.1 and 0.25 done on seeds 0-4; lambda=0.25 is the chosen setting
    #    (knee of the curve: ~same stability as 0.5, no collapsed seed) -> complete its seeds 5-9.
    "H_sst2_large_lambda":   dict(dataset="sst2", scale="large",
                                  conditions=["lar_pearson_l0.1", "lar_pearson_l0.25"]),
    # F: Rotten Tomatoes at full scale (8,530 training sentences x 2 epochs). lambda fixed to 0.25 BEFORE seeing this data.
    "F_rt_large":            dict(dataset="rotten_tomatoes", scale="large",
                                  conditions=["ce_only_control", "aug_only", "lar_aligned", "lar_pearson_l0.25"]),
    # G: IMDB at reduced scale (long reviews make 20k x 2 epochs impractical): 5,000 x 1 epoch. lambda = 0.25.
    "G_imdb_medium":         dict(dataset="imdb", scale="medium",
                                  conditions=["ce_only_control", "aug_only", "lar_aligned", "lar_pearson_l0.25"]),
    # E2: baselines (control, aug_only) re-run WITH the new metrics so the lambda variants have something to be compared to.
    #     Separate analysis group; the pooled "newmetrics" analysis pairs these with the H rows by seed.
    "E2_sst2_large_newmetrics": dict(dataset="sst2", scale="large", group="sst2_large_newmetrics",
                                  conditions=["ce_only_control", "aug_only"]),
}


def config_fingerprint(phase_name):
    """Everything that would change results if edited. Stored with the results; a mismatch on resume is an error."""
    p = PHASES[phase_name]
    payload = dict(protocol=PROTOCOL_VERSION, dataset=DATASETS[p["dataset"]], scale=SCALES[p["scale"]],
                   lr=LEARNING_RATE, train_strength=TRAIN_STRENGTH, eval_strengths=EVAL_STRENGTHS,
                   bank=BANK_PARAMS, ref=[REF_CLASSIFIER, REF_EMBEDDER], faith=[FAITH_FRACS, FAITH_N],
                   base=BASE_MODEL_NAME)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- results store
def _empty_store():
    return {"meta": {"protocol": PROTOCOL_VERSION, "phases": {}, "conditions": {}}, "rows": []}


def load_store():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            store = json.load(f)
        store.setdefault("meta", {}).setdefault("phases", {})
        store["meta"].setdefault("conditions", {})
        return store
    return _empty_store()


def save_store(store):
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=1)
    os.replace(tmp, RESULTS_JSON)
    pd.DataFrame(store["rows"]).to_csv(RESULTS_CSV, index=False)


def done_keys(store):
    """(phase, condition, seed) triples whose rows exist for ALL eval strengths."""
    seen = {}
    for r in store["rows"]:
        seen.setdefault((r["phase"], r["condition"], r["seed"]), set()).add(r["strength"])
    return {k for k, v in seen.items() if v == set(EVAL_STRENGTHS)}


# ---------------------------------------------------------------- restore from an attached input dataset
def _find_input_dir():
    hits = glob.glob("/kaggle/input/**/lar_v9_results.json", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    for cand in glob.glob("/kaggle/input/**/", recursive=True):
        if os.path.basename(os.path.normpath(cand)) == KAGGLE_DATASET_SLUG:
            return cand
    return None


def restore_from_input():
    src = _find_input_dir() if ON_KAGGLE else None
    if not src:
        print("No input dataset found -> starting fresh (expected on the very first run).")
        return
    n = 0
    for fname in os.listdir(src):
        path = os.path.join(src, fname)
        if fname in ("lar_v9_results.json", "lar_v9_results.csv"):
            dst = os.path.join(RUN_DIR, fname)
        elif fname.startswith("bank__") and fname.endswith(".json"):
            dst = os.path.join(BANK_DIR, fname)
        elif fname.startswith("ckpt__") and fname.endswith(".pt"):
            dst = os.path.join(CKPT_DIR, fname)
        else:
            continue
        if not os.path.exists(dst):
            shutil.copy2(path, dst)
            n += 1
            print("  restored", fname)
    print(f"Restore complete ({n} files) from {src}")


restore_from_input()

# ---------------------------------------------------------------- Kaggle API auto-commit (best effort)
_kaggle_ready = False


def _kaggle_credentials():
    global _kaggle_ready
    if _kaggle_ready:
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()
        user, key = s.get_secret("KAGGLE_USERNAME"), s.get_secret("KAGGLE_KEY")
        cdir = os.path.expanduser("~/.kaggle")
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, "kaggle.json"), "w") as f:
            json.dump({"username": user, "key": key}, f)
        os.chmod(os.path.join(cdir, "kaggle.json"), 0o600)
        _kaggle_ready = True
        return True
    except Exception as e:
        print(f"  [auto-commit] disabled ({e}). Local saves still work.")
        return False


def kaggle_auto_commit(message, include_ckpt=False):
    """Upload a FLAT staging folder (the Kaggle CLI skips sub-folders by default, so checkpoints must be
    top-level files). Results + banks are small; checkpoints (~0.8 GB each) only when include_ckpt=True."""
    if not (AUTO_COMMIT_TO_KAGGLE and ON_KAGGLE) or not _kaggle_credentials():
        return
    try:
        from kaggle_secrets import UserSecretsClient
        user = UserSecretsClient().get_secret("KAGGLE_USERNAME")
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
        os.makedirs(STAGE_DIR)
        for f in (RESULTS_JSON, RESULTS_CSV):
            if os.path.exists(f):
                shutil.copy2(f, STAGE_DIR)
        for f in glob.glob(os.path.join(BANK_DIR, "bank__*.json")):
            shutil.copy2(f, STAGE_DIR)
        if include_ckpt:
            for f in glob.glob(os.path.join(CKPT_DIR, "ckpt__*.pt")):
                shutil.copy2(f, STAGE_DIR)
        meta = {"title": KAGGLE_DATASET_TITLE, "id": f"{user}/{KAGGLE_DATASET_SLUG}",
                "licenses": [{"name": "CC0-1.0"}]}
        with open(os.path.join(STAGE_DIR, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f)
        r = subprocess.run(["kaggle", "datasets", "version", "-p", STAGE_DIR, "-m", message],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode == 0:
            print(f"  [auto-commit] pushed: {message}")
            return
        c = subprocess.run(["kaggle", "datasets", "create", "-p", STAGE_DIR],
                           capture_output=True, text=True, timeout=1800)
        print("  [auto-commit] created dataset (first commit)." if c.returncode == 0 else
              f"  [auto-commit] WARNING: version failed ({r.stderr[:150]}); create failed ({c.stderr[:150]})")
    except Exception as e:
        print(f"  [auto-commit] WARNING skipped: {e}")


_store = load_store()
print(f"Results store: {len(_store['rows'])} rows | completed (phase,cond,seed): {len(done_keys(_store))}")

# %%
# ===== CELL 2: data loading + validated perturbation banks (run every session; cached on disk) =====
import functools

import nltk
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

for _pkg in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass
from nltk.corpus import wordnet as wn  # noqa: E402


# ---------------------------------------------------------------- data
def _clean(t, truncate_words):
    t = t.replace("<br />", " ").strip()
    if truncate_words:
        t = " ".join(t.split()[:truncate_words])
    return t


def load_phase_data(dataset_key, scale_key):
    """Returns dict(train_texts, train_labels, eval_texts, eval_labels). Deterministic subsets."""
    from datasets import load_dataset
    cfg, sc = DATASETS[dataset_key], SCALES[scale_key]
    name, conf = cfg["hf"]
    args = (name,) if conf is None else (name, conf)

    tr = load_dataset(*args, split=cfg["train"])
    if cfg["shuffle_train"]:
        tr = tr.shuffle(seed=0)                      # imdb/rt train splits are label-sorted
    tr = tr.select(range(min(sc["train_size"], len(tr))))

    ev = load_dataset(*args, split=cfg["eval"])
    ev = ev.shuffle(seed=0)                          # fixed order for every phase/session
    n_eval = cfg["eval_n"] or len(ev)
    if sc["eval_limit"]:
        n_eval = min(n_eval, sc["eval_limit"])
    ev = ev.select(range(min(n_eval, len(ev))))

    tw = cfg["truncate_words"]
    return dict(train_texts=[_clean(t, tw) for t in tr[cfg["text"]]], train_labels=list(tr["label"]),
                eval_texts=[_clean(t, tw) for t in ev[cfg["text"]]], eval_labels=list(ev["label"]))


# ---------------------------------------------------------------- perturbation engine (deterministic)
_PUNCT_L, _PUNCT_R = "\"'(", ".,!?;:\"')"


def _wordnet_pos(tag):
    return wn.ADJ if tag.startswith("J") else (wn.ADV if tag.startswith("R") else None)


@functools.lru_cache(maxsize=None)
def _lemmas(clean_word, wpos):
    out = set()
    for syn in wn.synsets(clean_word, pos=wpos)[:BANK_PARAMS["max_synsets"]]:
        for lem in syn.lemmas():
            nm = lem.name().replace("_", " ")
            if nm.lower() != clean_word and nm.isalpha():       # single alphabetic word, different from original
                out.add(nm)
    return tuple(sorted(out))                                   # sorted => reproducible across processes


def _split_punct(w):
    lead = ""
    while w and w[0] in _PUNCT_L:
        lead, w = lead + w[0], w[1:]
    trail = ""
    while w and w[-1] in _PUNCT_R:
        trail, w = w[-1] + trail, w[:-1]
    return lead, w, trail


def find_candidates(words):
    cands = []
    for i, (w, tag) in enumerate(nltk.pos_tag(words)):
        wpos = _wordnet_pos(tag)
        if wpos is None:
            continue
        _, core, _ = _split_punct(w)
        if not core:
            continue
        lem = _lemmas(core.lower(), wpos)
        if lem:
            cands.append((i, lem))
    return cands


def apply_swaps(words, cands, strength, rng):
    """strength = fraction of eligible (adjective/adverb) words replaced; at least one, ceil as in the paper text."""
    k = len(cands)
    n_swaps = min(k, max(1, math.ceil(strength * k)))
    chosen = rng.sample(cands, n_swaps)
    new = list(words)
    for i, lem in chosen:
        lead, core, trail = _split_punct(words[i])
        repl = rng.choice(lem)
        if core.istitle():
            repl = repl.capitalize()
        new[i] = lead + repl + trail
    return dict(text=" ".join(new), n_swaps=n_swaps, k=k, nwords=len(words))


def _rng(dataset, idx, tag):
    return random.Random(f"{PROTOCOL_VERSION}|{dataset}|{idx}|{tag}")   # string seeds are process-independent


# ---------------------------------------------------------------- independent validity checks
class Validator:
    """Label-preservation (independent pretrained SST-2 classifier) + semantic similarity (MiniLM).
    Neither model is one of the models being evaluated, so acceptance cannot be tuned to inflate stability."""

    def __init__(self):
        self.ctok = AutoTokenizer.from_pretrained(REF_CLASSIFIER)
        self.clf = AutoModelForSequenceClassification.from_pretrained(REF_CLASSIFIER).to(device).eval()
        self.etok = AutoTokenizer.from_pretrained(REF_EMBEDDER)
        self.emb = AutoModel.from_pretrained(REF_EMBEDDER).to(device).eval()

    @torch.no_grad()
    def score(self, texts, bs=128, max_len=160):
        labels, embs = [], []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            e = self.ctok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            labels.append(self.clf(**e).logits.argmax(-1).cpu())
            e = self.etok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            h = self.emb(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            embs.append(F.normalize(v, dim=-1).cpu())
        return torch.cat(labels), torch.cat(embs)

    def free(self):
        del self.clf, self.emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_entries(validator, dataset, texts, tag, strength, max_draws, keep, chunk=256):
    """For every sentence draw up to `max_draws` distinct drafts, validate, keep the first `keep` that pass."""
    out = {}
    for start in range(0, len(texts), chunk):
        idxs = list(range(start, min(start + chunk, len(texts))))
        drafts = {}
        for i in idxs:
            words = texts[i].split()
            if not words:
                drafts[i] = []
                continue
            cands = find_candidates(words)
            rng, seen, lst = _rng(dataset, i, tag), set(), []
            if cands:
                for _ in range(max_draws * 3):
                    d = apply_swaps(words, cands, strength, rng)
                    if d["text"] != texts[i] and d["text"] not in seen:
                        seen.add(d["text"])
                        lst.append(d)
                    if len(lst) >= max_draws:
                        break
            drafts[i] = lst
        flat = [(i, d) for i in idxs for d in drafts[i]]
        all_texts = [texts[i] for i in idxs] + [d["text"] for _, d in flat]
        lab, emb = validator.score(all_texts)
        o_lab, o_emb = {i: lab[j] for j, i in enumerate(idxs)}, {i: emb[j] for j, i in enumerate(idxs)}
        acc = {i: [] for i in idxs}
        for j, (i, d) in enumerate(flat):
            k = len(idxs) + j
            sim = float((o_emb[i] * emb[k]).sum())
            if lab[k] == o_lab[i] and sim >= BANK_PARAMS["sim_threshold"] and len(acc[i]) < keep:
                acc[i].append(dict(text=d["text"], n_swaps=d["n_swaps"], k=d["k"], nwords=d["nwords"],
                                   sim=round(sim, 4)))
        for i in idxs:
            out[str(i)] = dict(n_candidates=len(drafts[i]) and drafts[i][0]["k"] or 0,
                               n_drafts=len(drafts[i]), variants=acc[i])
        print(f"    bank[{tag}] {min(start + chunk, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def _bank_path(kind, dataset, n):
    return os.path.join(BANK_DIR, f"bank__{kind}__{dataset}__n{n}.json")


def get_banks(dataset_key, scale_key, data):
    """Load or build (and persist) the train bank and the eval bank for this dataset/scale."""
    n_tr, n_ev = len(data["train_texts"]), len(data["eval_texts"])
    fp = hashlib.sha1(json.dumps([PROTOCOL_VERSION, BANK_PARAMS, REF_CLASSIFIER, REF_EMBEDDER,
                                  TRAIN_STRENGTH, EVAL_STRENGTHS], sort_keys=True).encode()).hexdigest()[:10]
    paths = {"train": _bank_path("train", dataset_key, n_tr), "eval": _bank_path("eval", dataset_key, n_ev)}
    banks, validator = {}, None
    for kind, texts in (("train", data["train_texts"]), ("eval", data["eval_texts"])):
        p = paths[kind]
        if os.path.exists(p):
            with open(p) as f:
                b = json.load(f)
            if b["meta"]["fingerprint"] != fp:
                raise RuntimeError(f"{p} was built with different bank settings. Delete it to rebuild.")
            banks[kind] = b
            print(f"  loaded {kind} bank ({len(b['entries'])} sentences) from disk")
            continue
        if validator is None:
            validator = Validator()
        print(f"  building {kind} bank for {dataset_key} (n={len(texts)}) ...")
        entries = {}
        if kind == "train":
            ent = _build_entries(validator, dataset_key, texts, "train", TRAIN_STRENGTH,
                                 BANK_PARAMS["max_draws_train"], BANK_PARAMS["train_variants"])
            entries = {i: {"train": e} for i, e in ent.items()}
        else:
            for sname, s in EVAL_STRENGTHS.items():
                ent = _build_entries(validator, dataset_key, texts, sname, s, BANK_PARAMS["max_draws_eval"], 1)
                for i, e in ent.items():
                    entries.setdefault(i, {})[sname] = e
        b = {"meta": {"fingerprint": fp, "protocol": PROTOCOL_VERSION, "dataset": dataset_key, "n": len(texts),
                      "params": BANK_PARAMS}, "entries": entries}
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(b, f)
        os.replace(tmp, p)
        banks[kind] = b
    if validator is not None:
        validator.free()
    return banks


def bank_report(banks, data=None):
    """Realised perturbation strength, acceptance rates and how often strengths coincide (issues #3, #4, #5)."""
    rows = []
    ev = banks["eval"]["entries"]
    n = len(ev)
    for sname in EVAL_STRENGTHS:
        acc = [e[sname]["variants"][0] for e in ev.values() if e[sname]["variants"]]
        has_c = sum(1 for e in ev.values() if e[sname]["n_drafts"] > 0)
        rows.append(dict(strength=sname, nominal=EVAL_STRENGTHS[sname], sentences=n,
                         with_candidates=has_c, accepted=len(acc), accept_rate_of_candidates=round(len(acc) / max(1, has_c), 3),
                         mean_swaps=round(float(np.mean([a["n_swaps"] for a in acc])), 2) if acc else None,
                         realised_frac_of_eligible=round(float(np.mean([a["n_swaps"] / a["k"] for a in acc])), 3) if acc else None,
                         realised_frac_of_words=round(float(np.mean([a["n_swaps"] / a["nwords"] for a in acc])), 3) if acc else None,
                         mean_sim=round(float(np.mean([a["sim"] for a in acc])), 3) if acc else None))
    df = pd.DataFrame(rows)
    same = {}
    for a, b in (("light", "medium"), ("medium", "heavy"), ("light", "heavy")):
        both = [(e[a]["variants"][0]["text"] == e[b]["variants"][0]["text"]) for e in ev.values()
                if e[a]["variants"] and e[b]["variants"]]
        same[f"{a}=={b}"] = round(float(np.mean(both)), 3) if both else None
    tr = banks["train"]["entries"]
    tr_acc = sum(1 for e in tr.values() if e["train"]["variants"])
    print(df.to_string(index=False))
    print("fraction of sentences whose perturbed text is IDENTICAL across strengths:", same)
    print(f"train bank: {tr_acc}/{len(tr)} sentences have >=1 validated variant "
          f"({100 * tr_acc / max(1, len(tr)):.1f}%); the rest are trained without a perturbation partner")
    return df, same

# %%
# ===== CELL 3: models, token alignment, stability + faithfulness evaluation =====


def load_fresh_model():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2).to(device)
    return model, tok


def word_emb(model):
    return model.distilbert.embeddings.word_embeddings


# ---------------------------------------------------------------- token alignment
def seq_align(ids_a, ids_b):
    """Order-preserving alignment of two token-id lists (difflib / Ratcliff-Obershelp matching blocks).
    Constant indices => usable inside a differentiable loss (we only GATHER attribution values)."""
    sm = difflib.SequenceMatcher(a=ids_a, b=ids_b, autojunk=False)
    ia, ib = [], []
    for blk in sm.get_matching_blocks():
        ia.extend(range(blk.a, blk.a + blk.size))
        ib.extend(range(blk.b, blk.b + blk.size))
    return ia, ib


def greedy_align(tokens_a, tokens_b):
    """v8 procedure (first not-yet-used identical token) - kept only for comparability with earlier results."""
    ma, mb, used = [], [], set()
    for oi, ot in enumerate(tokens_a):
        for pi, pt in enumerate(tokens_b):
            if pi not in used and pt == ot:
                ma.append(oi); mb.append(pi); used.add(pi)
                break
    return ma, mb


def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    r, _ = spearmanr(x, y)
    return float(r)


# ---------------------------------------------------------------- attribution (first-order input-gradient norm)
def attr_and_pred(model, tok, text, max_len, target=None):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    emb = word_emb(model)(enc["input_ids"])
    emb.requires_grad_(True)
    logits = model(inputs_embeds=emb, attention_mask=enc["attention_mask"]).logits[0]
    pred = int(logits.argmax())
    tgt = pred if target is None else int(target)
    grad = torch.autograd.grad(logits[tgt], emb)[0]
    attr = grad.norm(dim=-1).squeeze(0).detach().cpu().numpy()
    ids = enc["input_ids"][0].tolist()
    return ids, attr, pred


def ig_attr(model, tok, text, max_len, target):
    """Integrated Gradients (PAD-embedding baseline for content tokens, CLS/SEP kept), |sum over embedding dim|.
    A second, independent attribution method used only at evaluation time."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        emb = word_emb(model)(enc["input_ids"])
        base = emb.clone()
        base[:, 1:-1, :] = word_emb(model)(torch.tensor([[tok.pad_token_id]], device=device))
    alphas = torch.linspace(1.0 / IG_STEPS, 1.0, IG_STEPS, device=device).view(-1, 1, 1)
    x = (base + alphas * (emb - base)).detach().requires_grad_(True)           # [steps, L, D]
    mask = torch.ones(IG_STEPS, emb.shape[1], dtype=torch.long, device=device)
    logits = model(inputs_embeds=x, attention_mask=mask).logits[:, int(target)].sum()
    grads = torch.autograd.grad(logits, x)[0].mean(0, keepdim=True)            # [1, L, D]
    return ((emb - base) * grads).sum(-1).abs().squeeze(0).detach().cpu().numpy()


@torch.no_grad()
def accuracy_full(model, tok, texts, labels, max_len, bs=64):
    model.eval()
    correct = 0
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        pred = model(**enc).logits.argmax(-1).cpu()
        correct += int((pred == torch.tensor(labels[i:i + bs])).sum())
    return correct / max(1, len(texts))


# ---------------------------------------------------------------- faithfulness (comprehensiveness / sufficiency)
@torch.no_grad()
def _probs_of(model, ids_batch, pred):
    return F.softmax(model(input_ids=ids_batch).logits, dim=-1)[:, pred].cpu().numpy()


def faithfulness(model, tok, texts, labels, idx_subset, max_len):
    """ERASER-style comprehensiveness (drop in p(pred) when top-k% tokens are masked; higher = more faithful) and
    sufficiency (drop when ONLY top-k% tokens are kept; lower = more faithful), averaged over FAITH_FRACS,
    each compared to a random-ranking baseline so the numbers are interpretable."""
    model.eval()
    mask_id = tok.mask_token_id
    comp, suff, comp_r, suff_r = [], [], [], []
    for idx in idx_subset:
        ids, attr, pred = attr_and_pred(model, tok, texts[idx], max_len, target=None)
        n = len(ids) - 2
        if n < 2:
            continue
        content = np.arange(1, n + 1)
        order_attr = content[np.argsort(-attr[1:n + 1], kind="stable")]
        order_rand = content[np.random.RandomState(idx).permutation(n)]
        base = torch.tensor(ids, device=device).unsqueeze(0)
        variants, meta = [base], []
        for order, tag in ((order_attr, "a"), (order_rand, "r")):
            for f in FAITH_FRACS:
                k = max(1, int(round(f * n)))
                top = torch.tensor(order[:k], device=device)
                m_comp = base.clone(); m_comp[0, top] = mask_id                       # remove top-k
                m_suff = torch.full_like(base, mask_id); m_suff[0, [0, n + 1]] = base[0, [0, n + 1]]
                m_suff[0, top] = base[0, top]                                       # keep only top-k
                variants += [m_comp, m_suff]
                meta.append(tag)
        probs = _probs_of(model, torch.cat(variants, 0), pred)
        p0, rest = probs[0], probs[1:].reshape(-1, 2)
        for (pc, ps), tag in zip(rest, meta):
            (comp if tag == "a" else comp_r).append(p0 - pc)
            (suff if tag == "a" else suff_r).append(p0 - ps)
    m = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(comp=m(comp), suff=m(suff), comp_rand=m(comp_r), suff_rand=m(suff_r),
                comp_gain=m(comp) - m(comp_r), suff_gain=m(suff_r) - m(suff), faith_n=len(idx_subset))


# ---------------------------------------------------------------- one-pass evaluation of a trained model
def evaluate_model(model, tok, data, bank_eval, max_len):
    model.eval()
    texts, labels = data["eval_texts"], data["eval_labels"]
    out = {"acc_full": accuracy_full(model, tok, texts, labels, max_len)}

    orig = {}                                                         # attribution of originals, computed once
    for i, t in enumerate(texts):
        orig[i] = attr_and_pred(model, tok, t, max_len, target=labels[i])

    per_strength = {}
    for sname in EVAL_STRENGTHS:
        r_seq, r_greedy, r_pos, cover, flips, pert_correct, n_pairs = [], [], [], [], [], 0, 0
        r_pear, jac, ig_rho = [], [], []
        for i in range(len(texts)):
            var = bank_eval["entries"][str(i)][sname]["variants"]
            if not var:
                continue
            ids_o, a_o, pred_o = orig[i]
            ids_p, a_p, pred_p = attr_and_pred(model, tok, var[0]["text"], max_len, target=labels[i])
            n_pairs += 1
            flips.append(int(pred_o != pred_p))
            pert_correct += int(pred_p == labels[i])
            co, cp = ids_o[1:-1], ids_p[1:-1]                         # content tokens (no CLS/SEP)
            ao, ap = a_o[1:-1], a_p[1:-1]
            ia, ib = seq_align(co, cp)
            if len(ia) >= 3:
                r = _spearman(ao[ia], ap[ib])
                if not math.isnan(r):
                    r_seq.append(r)
                    cover.append(len(ia) / max(1, len(co)))
                    if np.std(ao[ia]) > 0 and np.std(ap[ib]) > 0:               # loss-matched metric (reported, not primary)
                        r_pear.append(float(np.corrcoef(ao[ia], ap[ib])[0, 1]))
                    k = max(1, int(round(TOPK_FRAC * len(ia))))                  # metric NOT optimised by any loss
                    top_o = set(np.argsort(-ao[ia], kind="stable")[:k].tolist())
                    top_p = set(np.argsort(-ap[ib], kind="stable")[:k].tolist())
                    jac.append(len(top_o & top_p) / len(top_o | top_p))
                    if sname == "medium" and len(ig_rho) < IG_N:                 # second attribution method
                        g_o = ig_attr(model, tok, texts[i], max_len, labels[i])[1:-1]
                        g_p = ig_attr(model, tok, var[0]["text"], max_len, labels[i])[1:-1]
                        rg = _spearman(g_o[ia], g_p[ib])
                        if not math.isnan(rg):
                            ig_rho.append(rg)
            ga, gb = greedy_align(co, cp)
            if len(ga) >= 3:
                r = _spearman(ao[ga], ap[gb])
                if not math.isnan(r):
                    r_greedy.append(r)
            L = min(len(co), len(cp))
            if L >= 3:
                r = _spearman(ao[:L], ap[:L])                         # naive positional (the artifact)
                if not math.isnan(r):
                    r_pos.append(r)
        mean = lambda v: float(np.mean(v)) if v else float("nan")
        per_strength[sname] = dict(rho_seq=mean(r_seq), rho_greedy=mean(r_greedy), rho_positional=mean(r_pos),
                                   rho_pearson=mean(r_pear), topk_jaccard=mean(jac), ig_rho=mean(ig_rho),
                                   coverage=mean(cover), n_valid_seq=len(r_seq), n_pairs=n_pairs,
                                   flip_rate=mean(flips), acc_pert=pert_correct / max(1, n_pairs))
    rs = random.Random(12345)
    subset = sorted(rs.sample(range(len(texts)), min(FAITH_N, len(texts))))
    out.update(faithfulness(model, tok, texts, labels, subset, max_len))
    return out, per_strength

# %%
# ===== CELL 4: losses, resumable training, phase runner =====
from torch.optim import AdamW


@contextlib.contextmanager
def math_sdpa():
    """Flash / mem-efficient attention kernels have no double-backward; force the math backend for LAR losses."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    with ctx:
        yield


def _attr_with_graph(model, ids, labels):
    emb = word_emb(model)(ids)
    logits = model(inputs_embeds=emb).logits
    tl = logits.gather(1, labels.unsqueeze(-1)).squeeze(-1)
    g = torch.autograd.grad(tl.sum(), emb, create_graph=True, retain_graph=True)[0]
    return logits, g.norm(dim=-1)                                     # [B, L]


def _distance(a, b, dist):
    if dist == "pearson":
        a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8).mean()


def compute_loss(model, cond, ids_o, ids_p, labels):
    """ids_p is None when a sentence has no validated perturbation (then the partner term is skipped)."""
    kind = cond["kind"]
    stats = dict(ce=0.0, reg=0.0, matched=-1)
    if kind == "ce" or ids_p is None and kind != "lar":
        loss = F.cross_entropy(model(input_ids=ids_o).logits, labels)
        stats["ce"] = loss.item()
        return loss, stats
    if kind == "aug":
        lo, lp = model(input_ids=ids_o).logits, model(input_ids=ids_p).logits
        loss = 0.5 * (F.cross_entropy(lo, labels) + F.cross_entropy(lp, labels))
        stats["ce"] = loss.item()
        return loss, stats
    if kind == "kl":
        lo, lp = model(input_ids=ids_o).logits, model(input_ids=ids_p).logits
        ce = F.cross_entropy(lo, labels)
        lpo, lpp = F.log_softmax(lo, -1), F.log_softmax(lp, -1)
        kl = 0.5 * (F.kl_div(lpp, lpo, log_target=True, reduction="batchmean")
                    + F.kl_div(lpo, lpp, log_target=True, reduction="batchmean"))
        loss = ce + cond["lam"] * kl
        stats.update(ce=ce.item(), reg=kl.item())
        return loss, stats
    # ---- LAR family (double backward)
    with math_sdpa():
        if ids_p is None:
            lo = model(input_ids=ids_o).logits
            loss = F.cross_entropy(lo, labels)
            stats["ce"] = loss.item()
            return loss, stats
        lo, a_o = _attr_with_graph(model, ids_o, labels)
        _, a_p = _attr_with_graph(model, ids_p, labels)
        ce = F.cross_entropy(lo, labels)
        if cond["align"] == "positional":                             # exactly the v8 loss
            a_o = a_o / (a_o.norm(dim=-1, keepdim=True) + 1e-8)
            a_p = a_p / (a_p.norm(dim=-1, keepdim=True) + 1e-8)
            L = min(a_o.shape[1], a_p.shape[1])
            reg = _distance(a_o[:, :L], a_p[:, :L], "cos")
        else:                                                         # alignment-aware (gather matched content tokens)
            ia, ib = seq_align(ids_o[0, 1:-1].tolist(), ids_p[0, 1:-1].tolist())
            stats["matched"] = len(ia)
            if len(ia) >= 3:
                ia_t = torch.tensor(ia, device=ids_o.device) + 1
                ib_t = torch.tensor(ib, device=ids_o.device) + 1
                reg = _distance(a_o[:, ia_t], a_p[:, ib_t], cond["dist"])
            else:
                reg = ce * 0.0
        loss = ce + cond["lam"] * reg
        stats.update(ce=ce.item(), reg=float(reg.detach()))
        return loss, stats


# ---------------------------------------------------------------- checkpoints (flat files, atomic writes)
def ckpt_path(phase, cond, seed):
    return os.path.join(CKPT_DIR, f"ckpt__{phase}__{cond}__seed{seed}.pt")


def save_ckpt(phase, cond, seed, step, model, opt, rng):
    p = ckpt_path(phase, cond, seed)
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, p + ".tmp")
    os.replace(p + ".tmp", p)


def load_ckpt(phase, cond, seed, model, opt, rng):
    p = ckpt_path(phase, cond, seed)
    if not os.path.exists(p):
        return 0
    ck = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); rng.setstate(ck["rng"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    if ck["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
    print(f"  resumed {phase}/{cond}/seed{seed} from step {ck['step']}")
    return ck["step"]


def drop_ckpt(phase, cond, seed):
    p = ckpt_path(phase, cond, seed)
    if os.path.exists(p):
        os.remove(p)


# ---------------------------------------------------------------- training
def train_one(phase, cond_name, seed, data, banks, ds_cfg, sc_cfg):
    cond = CONDITIONS[cond_name]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model, tok = load_fresh_model()
    opt = AdamW(model.parameters(), lr=LEARNING_RATE)
    rng = random.Random(seed)                                         # picks the perturbation variant per step
    start = load_ckpt(phase, cond_name, seed, model, opt, rng)

    n = len(data["train_texts"])
    seq = []
    srng = random.Random(seed)
    for _ in range(sc_cfg["epochs"]):
        idxs = list(range(n)); srng.shuffle(idxs); seq.extend(idxs)
    total = len(seq)
    if start >= total:
        print(f"  already fully trained ({total} steps)")
        model.eval()
        return model, tok

    model.train()
    ce_acc = reg_acc = 0.0
    t0 = time.time()
    for step in range(start, total):
        i = seq[step]
        variants = banks["train"]["entries"][str(i)]["train"]["variants"]
        pick = rng.randrange(max(1, len(variants)))                   # always consume the RNG (aligned streams)
        pert = variants[pick]["text"] if variants else None
        enc_o = tok(data["train_texts"][i], return_tensors="pt", truncation=True, max_length=ds_cfg["max_len"]).to(device)
        ids_o = enc_o["input_ids"]
        ids_p = None
        if pert is not None:
            ids_p = tok(pert, return_tensors="pt", truncation=True, max_length=ds_cfg["max_len"])["input_ids"].to(device)
        labels = torch.tensor([data["train_labels"][i]], device=device)
        opt.zero_grad(set_to_none=True)
        loss, st = compute_loss(model, cond, ids_o, ids_p, labels)
        loss.backward()
        opt.step()
        ce_acc += st["ce"]; reg_acc += st["reg"]
        s1 = step + 1
        if s1 % LOG_EVERY == 0:
            k = LOG_EVERY
            print(f"  [{phase}|{cond_name}|seed{seed}] {s1}/{total} ce={ce_acc / k:.4f} reg={reg_acc / k:.4f} "
                  f"({(time.time() - t0) / (s1 - start):.2f}s/step)")
            ce_acc = reg_acc = 0.0
        if s1 % sc_cfg["ckpt_every"] == 0 and s1 < total:
            save_ckpt(phase, cond_name, seed, s1, model, opt, rng)
            if sc_cfg["commit_ckpt_every"] and s1 % sc_cfg["commit_ckpt_every"] == 0:
                kaggle_auto_commit(f"{phase}/{cond_name}/seed{seed} step {s1}/{total}", include_ckpt=True)
    save_ckpt(phase, cond_name, seed, total, model, opt, rng)         # kept until results rows are saved
    model.eval()
    return model, tok


# ---------------------------------------------------------------- phase runner
def run_phase(phase_name, conditions=None, seeds=None):
    """Resumable: skips (condition, seed) pairs already stored, resumes a half-trained model from its checkpoint.
    Seed-outer loop => after k seeds you already have paired data for every condition."""
    global _store
    p = PHASES[phase_name]
    conds = conditions or p["conditions"]
    seeds = SEEDS if seeds is None else seeds
    ds_cfg, sc_cfg = DATASETS[p["dataset"]], SCALES[p["scale"]]
    store = load_store()
    fp = config_fingerprint(phase_name)
    prev = store["meta"]["phases"].get(phase_name)
    if prev and prev["fingerprint"] != fp:
        raise RuntimeError(f"Config for {phase_name} changed since earlier results were written "
                           f"({prev['fingerprint']} -> {fp}). Revert the change or rename the phase.")
    store["meta"]["phases"][phase_name] = {"fingerprint": fp, "dataset": p["dataset"], "scale": p["scale"]}
    for c in conds:
        old = store["meta"]["conditions"].get(c)
        if old and old != CONDITIONS[c]:
            raise RuntimeError(f"Definition of condition '{c}' changed since earlier results were written.")
        store["meta"]["conditions"][c] = CONDITIONS[c]
    save_store(store)

    todo = [(s, c) for s in seeds for c in conds if (phase_name, c, s) not in done_keys(store)]
    print(f"== {phase_name}: {len(conds) * len(seeds) - len(todo)} done, {len(todo)} to run ==")
    if not todo:
        return
    data = load_phase_data(p["dataset"], p["scale"])
    banks = get_banks(p["dataset"], p["scale"], data)
    bank_report(banks)
    kaggle_auto_commit(f"{phase_name}: banks ready")

    try:
        for k, (seed, cond_name) in enumerate(todo, 1):
            print(f"\n=== [{k}/{len(todo)}] {phase_name} | {cond_name} | seed {seed} ===")
            t0 = time.time()
            model, tok = train_one(phase_name, cond_name, seed, data, banks, ds_cfg, sc_cfg)
            t_train = time.time() - t0
            model_metrics, per_strength = evaluate_model(model, tok, data, banks["eval"], ds_cfg["max_len"])
            print(f"  acc_full={model_metrics['acc_full']:.4f} | comp_gain={model_metrics['comp_gain']:.4f} "
                  f"suff_gain={model_metrics['suff_gain']:.4f} | " +
                  " ".join(f"{s}:rho={v['rho_seq']:.4f}" for s, v in per_strength.items()))
            store = load_store()
            for sname, v in per_strength.items():
                store["rows"].append(dict(phase=phase_name, dataset=p["dataset"], scale=p["scale"],
                                          group=p.get("group", f"{p['dataset']}_{p['scale']}"), condition=cond_name,
                                          seed=seed, strength=sname, protocol=PROTOCOL_VERSION, config_hash=fp,
                                          train_size=sc_cfg["train_size"], epochs=sc_cfg["epochs"],
                                          train_seconds=round(t_train, 1), **model_metrics, **v))
            save_store(store)
            drop_ckpt(phase_name, cond_name, seed)
            kaggle_auto_commit(f"{phase_name}/{cond_name}/seed{seed} complete")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except BaseException as e:
        print(f"\n!!! interrupted ({type(e).__name__}: {e}) - results so far are saved; committing then re-raising")
        kaggle_auto_commit(f"{phase_name}: interrupted", include_ckpt=bool(sc_cfg["commit_ckpt_every"]))
        raise
    print(f"\n{phase_name}: finished.")


def reset_phase(phase_name):
    """Delete stored rows + checkpoints of one phase (use for S_smoke, or to deliberately redo a phase)."""
    store = load_store()
    store["rows"] = [r for r in store["rows"] if r["phase"] != phase_name]
    store["meta"]["phases"].pop(phase_name, None)
    save_store(store)
    for f in glob.glob(os.path.join(CKPT_DIR, f"ckpt__{phase_name}__*")):
        os.remove(f)
    print(f"reset {phase_name}")

# %%
# ===== CELL 5: analysis (re-runnable at any time; uses whatever seeds have finished) =====
from scipy import stats as sst

ANALYSIS_DIR = os.path.join(RUN_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)
HOLM_FAMILY = ["lar_pos", "lar_aligned", "lar_pearson", "pred_consistency"]   # each tested vs aug_only (primary family)
REF_FOR_AUG = "aug_only"
MODEL_METRICS = ["acc_full", "comp_gain", "suff_gain"]


def load_df(include_smoke=False):
    """All rows. `group` = analysis group (default dataset_scale); rows written before v9.1 get it filled in."""
    df = pd.DataFrame(load_store()["rows"])
    if df.empty:
        return df
    if not include_smoke:
        df = df[~df.phase.str.startswith("S_")]
    if "group" not in df:
        df["group"] = np.nan
    df["group"] = df["group"].fillna(df["dataset"] + "_" + df["scale"])
    for m in ("topk_jaccard", "rho_pearson", "ig_rho"):
        if m not in df:
            df[m] = np.nan
    key = ["group", "condition", "seed", "strength"]
    dup = df.duplicated(key, keep="first")
    if dup.any():
        print(f"WARNING: {int(dup.sum())} duplicate rows within a group dropped (keep first)")
    return df[~dup]


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired_contrast(df, metric, treat, ref, group_cols, n_boot=10000, seed=0):
    """Paired-by-seed contrast of `treat` minus `ref` for one metric inside each group.
    NOTE: with 10 seeds the percentile-bootstrap CI is slightly optimistic; prefer p_t / the t-interval when they disagree."""
    rows = []
    for g, sub in df.groupby(group_cols):
        w = sub.pivot_table(index="seed", columns="condition", values=metric)
        if treat not in w or ref not in w:
            continue
        d = (w[treat] - w[ref]).dropna()
        if len(d) < 3:
            continue
        rng = np.random.RandomState(seed)
        boots = [d.values[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
        sd = d.std(ddof=1)
        t = sst.ttest_rel(w.loc[d.index, treat], w.loc[d.index, ref])
        tci = sst.t.interval(0.95, len(d) - 1, d.mean(), sd / np.sqrt(len(d))) if sd > 0 else (np.nan, np.nan)
        try:
            wp = sst.wilcoxon(d).pvalue
        except ValueError:
            wp = float("nan")
        g = g if isinstance(g, tuple) else (g,)
        rows.append(dict(zip(group_cols, g), metric=metric, treat=treat, ref=ref, n_seeds=len(d),
                         mean_diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                         t_ci_lo=tci[0], t_ci_hi=tci[1],
                         d_z=d.mean() / sd if sd > 0 else float("nan"), p_t=t.pvalue, p_wilcoxon=wp,
                         n_pos=int((d > 0).sum())))
    return pd.DataFrame(rows)


def _cat(frames):
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_analysis(verbose=True):
    df = load_df()
    if df.empty:
        print("no results yet"); return None
    out = {}
    sg, mg = ["group", "strength"], ["group"]
    model_df = df.drop_duplicates(["group", "condition", "seed"])
    present = set(df.condition)
    treatments = [c for c in HOLM_FAMILY if c in present] + sorted(c for c in present if c.startswith("lar_pearson_l"))

    # 1) primary family (Holm within group/strength): stability of each treatment vs aug_only
    prim = _cat([paired_contrast(df, "rho_seq", tr, REF_FOR_AUG, sg) for tr in treatments])
    if len(prim):
        prim["p_holm"] = np.nan
        for _, idx in prim.groupby(sg).groups.items():
            prim.loc[idx, "p_holm"] = holm(prim.loc[idx, "p_t"].values)
    out["primary_vs_aug"] = prim

    # 2) metrics NOT matched to a training loss (top-k overlap) + loss-matched Pearson + IG stability (medium only)
    out["secondary_metrics_vs_aug"] = _cat([paired_contrast(df, m, tr, REF_FOR_AUG, sg)
                                            for m in ["topk_jaccard", "rho_pearson", "ig_rho"] for tr in treatments])

    # 2b) pooled "new metrics" analysis: rows that HAVE the new metrics, from any group of the same dataset/scale,
    #     paired by seed (e.g. H lambda rows vs E2 baselines). Never mixes models that lack the metrics.
    nm = df[df.topk_jaccard.notna()].copy()
    if len(nm):
        nm["group"] = nm["dataset"] + "_" + nm["scale"] + "_nm"
        nm_treat = sorted(c for c in set(nm.condition) if c not in ("ce_only_control", "aug_only"))
        nm_model = nm.drop_duplicates(["group", "condition", "seed"])
        parts = [paired_contrast(nm, m, tr, REF_FOR_AUG, sg) for m in ["rho_seq", "topk_jaccard", "rho_pearson", "ig_rho"] for tr in nm_treat]
        parts += [paired_contrast(nm, m, "ce_only_control", REF_FOR_AUG, sg) for m in ["rho_seq", "topk_jaccard", "ig_rho"]]
        parts += [paired_contrast(nm_model, m, tr, REF_FOR_AUG, mg) for m in MODEL_METRICS for tr in nm_treat]
        out["newmetrics_vs_aug"] = _cat(parts)

    # 3) context contrasts (descriptive)
    out["context"] = _cat([paired_contrast(df, "rho_seq", a, b, sg) for a, b in
                           [("lar_pos", "ce_only_control"), ("aug_only", "ce_only_control"),
                            ("lar_aligned", "lar_pos"), ("lar_pearson", "lar_aligned"), ("lar_pearson", "lar_pos")]])

    # 4) accuracy / faithfulness side effects (model-level) vs aug_only; accuracy also vs control
    side = [paired_contrast(model_df, m, tr, REF_FOR_AUG, mg) for m in MODEL_METRICS for tr in treatments + ["ce_only_control"]]
    side += [paired_contrast(model_df, "acc_full", tr, "ce_only_control", mg) for tr in treatments]
    out["side_effects"] = _cat(side)

    # 5) controlled alignment-artifact ablation: same models, three alignment procedures
    out["alignment_ablation"] = df.groupby(["group", "condition", "strength"])[
        ["rho_positional", "rho_greedy", "rho_seq", "coverage"]].mean().reset_index()

    # 6) descriptive table
    out["descriptive"] = df.groupby(["group", "condition", "strength"])[
        ["rho_seq", "topk_jaccard", "rho_pearson", "ig_rho"]].agg(["mean", "std", "count"])

    for k, v in out.items():
        if len(v):
            v.to_csv(os.path.join(ANALYSIS_DIR, f"{k}.csv"))
    if verbose:
        pd.set_option("display.width", 240, "display.max_columns", 40)
        for k in ["primary_vs_aug", "newmetrics_vs_aug", "secondary_metrics_vs_aug", "context", "side_effects", "alignment_ablation"]:
            if k not in out:
                continue
            print(f"\n----- {k} -----")
            print(out[k].round(4).to_string(index=False) if len(out[k]) else "(nothing yet)")
        print(f"\nCSV files written to {ANALYSIS_DIR}")
    return out


def tradeoff_report(group="sst2_large", seeds="common"):
    """Stability gain versus accuracy cost per condition (medium strength) - use it to choose lambda.
    seeds="common" (default): only seeds present for EVERY condition, so means are paired and comparable;
    seeds="all": every available seed (means then rest on different seeds - do not read differences from it);
    or pass an explicit list of seeds."""
    df = load_df()
    df = df[df.group == group]
    if df.empty:
        print("no rows for", group); return None
    per_cond = df.groupby("condition").seed.apply(set)
    if seeds == "common":
        use = set.intersection(*per_cond.values)
    elif seeds == "all":
        use = set(df.seed)
    else:
        use = set(seeds)
    df = df[df.seed.isin(use)]
    print(f"seeds used: {sorted(use)}  | seeds available per condition: " + ", ".join(f"{c}={len(s)}" for c, s in per_cond.items()))
    m = df.drop_duplicates(["condition", "seed"]).set_index(["condition", "seed"])
    med = df[df.strength == "medium"].set_index(["condition", "seed"])
    t = pd.DataFrame({"rho_medium": med.rho_seq, "topk_jaccard": med.topk_jaccard, "ig_rho": med.ig_rho,
                      "acc_full": m.acc_full, "comp_gain": m.comp_gain}).groupby("condition").agg(["mean", "std"])
    if REF_FOR_AUG in t.index:
        t[("acc_vs_aug", "mean")] = t[("acc_full", "mean")] - t.loc[REF_FOR_AUG, ("acc_full", "mean")]
        t[("rho_vs_aug", "mean")] = t[("rho_medium", "mean")] - t.loc[REF_FOR_AUG, ("rho_medium", "mean")]
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print(t.round(4).sort_values(("rho_medium", "mean")).to_string())
    return t


def gate_report():
    """Stage-0 decision at small scale: does LAR still beat the augmentation-only control?"""
    df = load_df()
    df = df[df.group == "sst2_small"]
    if df.empty:
        print("phase A not run yet"); return
    for tr in ["lar_pos", "aug_only"]:
        ref = "aug_only" if tr == "lar_pos" else "ce_only_control"
        c = paired_contrast(df, "rho_seq", tr, ref, ["strength"])
        print(f"\n{tr} - {ref}  (rho_seq, paired by seed)")
        print(c[["strength", "n_seeds", "mean_diff", "ci_lo", "ci_hi", "d_z", "p_t", "p_wilcoxon", "n_pos"]].round(4).to_string(index=False))


def latex_main_table(group="sst2_large", conditions=None, path=None):
    """Mean (sd) of rho_seq per condition x strength, plus Holm-adjusted p vs aug_only, as a booktabs table."""
    df = load_df()
    df = df[df.group == group]
    conditions = conditions or [c for c in CONDITIONS if c in set(df.condition)]
    prim = run_analysis(verbose=False)["primary_vs_aug"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Condition & Light & Medium & Heavy \\", r"\midrule"]
    for c in conditions:
        cells = []
        for s in EVAL_STRENGTHS:
            v = df[(df.condition == c) & (df.strength == s)].rho_seq
            cell = f"{v.mean():.4f} ({v.std():.4f})" if len(v) else "--"
            if len(prim):
                m = prim[(prim.group == group) & (prim.strength == s) & (prim.treat == c)]
                if len(m) and m.p_holm.iloc[0] < 0.05:
                    cell += r"$^{\dagger}$"
            cells.append(cell)
        name = c.replace("_", "\\_")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% $\dagger$: Holm-adjusted paired-$t$ p<0.05 versus aug\_only"]
    tex = "\n".join(lines)
    path = path or os.path.join(ANALYSIS_DIR, f"table_{group}.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(tex)
    return tex


def export_bundle():
    """Zip results + analysis + banks into one file (download from the Output tab or attach to the paper repo)."""
    base = os.path.join(WORK, "lar_v9_export")
    shutil.make_archive(base, "zip", RUN_DIR, ".")
    print("wrote", base + ".zip")

# %%
# ===== CELL 6: dataset statistics for the paper's "Datasets" section (run after cells 1-5; needs internet for HF data) =====
DS_STAT_COMBOS = [("sst2", "large"), ("rotten_tomatoes", "large"), ("imdb", "medium")]


def _describe(vals):
    v = np.asarray(vals, dtype=float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))


def _split_stats(texts, labels, tok, max_len):
    words = [len(t.split()) for t in texts]
    ntok = [len(tok(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts]
    ks = [len(find_candidates(t.split())) if t.split() else 0 for t in texts]
    return dict(n=len(texts), positive_frac=float(np.mean(labels)), words=_describe(words), wordpieces=_describe(ntok),
                truncated_frac=float(np.mean([n > max_len for n in ntok])),
                candidates_mean=float(np.mean(ks)), with_candidate_frac=float(np.mean([k > 0 for k in ks])))


def _bank_stats(dataset, n_train, n_eval):
    out = {}
    p_tr, p_ev = (os.path.join(BANK_DIR, f"bank__{kind}__{dataset}__n{n}.json") for kind, n in (("train", n_train), ("eval", n_eval)))
    if os.path.exists(p_tr):
        T = json.load(open(p_tr))["entries"]
        acc = [e["train"]["variants"][0] for e in T.values() if e["train"]["variants"]]
        out["train"] = dict(with_validated_variant_frac=len(acc) / max(1, len(T)),
                            with_two_variants_frac=float(np.mean([len(e["train"]["variants"]) >= 2 for e in T.values()])),
                            mean_swaps=float(np.mean([a["n_swaps"] for a in acc])) if acc else None,
                            mean_similarity=float(np.mean([a["sim"] for a in acc])) if acc else None)
    if os.path.exists(p_ev):
        E = json.load(open(p_ev))["entries"]
        out["eval"] = {}
        for s in EVAL_STRENGTHS:
            has_c = sum(1 for e in E.values() if e[s]["n_drafts"] > 0)
            acc = [e[s]["variants"][0] for e in E.values() if e[s]["variants"]]
            out["eval"][s] = dict(with_candidates=has_c, accepted=len(acc), acceptance_of_candidates=len(acc) / max(1, has_c),
                                  mean_swaps=float(np.mean([a["n_swaps"] for a in acc])) if acc else None,
                                  realised_frac_of_eligible=float(np.mean([a["n_swaps"] / a["k"] for a in acc])) if acc else None,
                                  realised_frac_of_words=float(np.mean([a["n_swaps"] / a["nwords"] for a in acc])) if acc else None,
                                  mean_similarity=float(np.mean([a["sim"] for a in acc])) if acc else None)
    return out


def dataset_statistics(combos=None, tok=None, save=True):
    """Per (dataset, scale): statistics of the training subset actually used and of the evaluation set, plus the
    perturbation-bank acceptance numbers if the banks are on disk. Writes dataset_stats.json and tab_datasets.tex."""
    tok = tok or AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    stats = {}
    for dataset, scale in (combos or DS_STAT_COMBOS):
        print(f"--- {dataset} / {scale}")
        data = load_phase_data(dataset, scale)
        ml = DATASETS[dataset]["max_len"]
        stats[f"{dataset}_{scale}"] = dict(
            dataset=dataset, scale=scale, max_len=ml, train_used=_split_stats(data["train_texts"], data["train_labels"], tok, ml),
            eval=_split_stats(data["eval_texts"], data["eval_labels"], tok, ml),
            banks=_bank_stats(dataset, len(data["train_texts"]), len(data["eval_texts"])))
    if save:
        with open(os.path.join(RUN_DIR, "dataset_stats.json"), "w") as f:
            json.dump(stats, f, indent=1)
        rows = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
                r"Dataset & Split & $n$ & Pos.\ (\%) & Words (mean/med.) & Word-pieces (mean/p90) & $>$max len (\%) & Cand.\ / ex. & With cand.\ (\%) \\", r"\midrule"]
        for key, d in stats.items():
            for split, label in (("train_used", "train (used)"), ("eval", "eval")):
                s = d[split]
                rows.append(f"{d['dataset'].replace('_', ' ')} ({d['scale']}) & {label} & {s['n']:,} & {100 * s['positive_frac']:.1f} & "
                            f"{s['words']['mean']:.1f}/{s['words']['median']:.0f} & {s['wordpieces']['mean']:.1f}/{s['wordpieces']['p90']:.0f} & "
                            f"{100 * s['truncated_frac']:.1f} & {s['candidates_mean']:.2f} & {100 * s['with_candidate_frac']:.1f} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(RUN_DIR, "tab_datasets.tex"), "w") as f:
            f.write("\n".join(rows))
        print("wrote dataset_stats.json and tab_datasets.tex to", RUN_DIR)
    return stats

# %% [markdown]
# ## Already completed (kept for reference — skipped automatically when their rows exist)
# S (smoke), A (SST-2 small core), B (SST-2 small variants), E (SST-2 large, 5 conditions × 10 seeds), H on seeds 0–4 (`lar_pearson_l0.1`, `lar_pearson_l0.25`).

# %%
# run_phase("A_sst2_small_core")
# run_phase("B_sst2_small_variants")
# run_phase("E_sst2_large")
# run_phase("H_sst2_large_lambda", seeds=[0, 1, 2, 3, 4])

# %% [markdown]
# ## Phase H (continued) — finish λ = 0.25 on seeds 5–9
# λ = 0.25 is the chosen setting, so it needs the full 10 seeds (≈ 46 min per run ≈ 3.8 GPU-hours). λ = 0.1 stays at 5 seeds as a sensitivity point (run `conditions=["lar_pearson_l0.1"], seeds=[5,6,7,8,9]` only if you want a 10-seed dose-response).

# %%
run_phase("H_sst2_large_lambda", conditions=["lar_pearson_l0.25"], seeds=[5, 6, 7, 8, 9])

# %% [markdown]
# ## Phase E2 — baselines with the new metrics (SST-2 large, seeds 0–4)
# Needed so `topk_jaccard` and `ig_rho` of the λ variants can be compared with control / aug_only. ≈ 29 min per run, ≈ 4.8 GPU-hours for 5 seeds × 2 conditions. Extend to seeds 5–9 after the λ = 0.25 seeds 5–9 finish if you want 10 paired seeds.

# %%
run_phase("E2_sst2_large_newmetrics", seeds=[0, 1, 2, 3, 4])
# run_phase("E2_sst2_large_newmetrics", seeds=[5, 6, 7, 8, 9])

# %% [markdown]
# ## Phase F — Rotten Tomatoes, full scale (8,530 sentences × 2 epochs ≈ 17k steps)
# Conditions: control, aug_only, `lar_aligned`, `lar_pearson_l0.25`. Estimated ≈ 85 min per seed for the four conditions (extrapolated from SST-2 timings — read s/step in the first run). Run seeds 0–4 first.

# %%
run_phase("F_rt_large", seeds=[0, 1, 2, 3, 4])
# run_phase("F_rt_large", seeds=[5, 6, 7, 8, 9])

# %% [markdown]
# ## Phase G — IMDB, reduced scale (5,000 × 1 epoch; reviews truncated to 90 words / 128 tokens)
# 20,000 × 2 epochs would need ≈ 40+ GPU-hours per seed (extrapolated), so IMDB uses a reduced budget. Read s/step from the first run before committing to all seeds.

# %%
run_phase("G_imdb_medium", seeds=[0, 1, 2, 3, 4])
# run_phase("G_imdb_medium", seeds=[5, 6, 7, 8, 9])

# %% [markdown]
# ## Dataset statistics for the paper (run once; needs the datasets and the banks on disk)
# Run after the banks of the phases you care about exist (banks are built the first time a phase runs; the SST-2 banks are restored from your dataset).

# %%
ds_stats = dataset_statistics()

# %% [markdown]
# ## Analysis, tables, export
# Re-runnable at any time on whatever seeds have finished.

# %%
out = run_analysis()

# %%
tradeoff_report("sst2_large")                 # common seeds only (paired)
# tradeoff_report("rotten_tomatoes_large"); tradeoff_report("imdb_medium")

# %%
latex_main_table("sst2_large")
# latex_main_table("rotten_tomatoes_large"); latex_main_table("imdb_medium"); latex_main_table("sst2_small")

# %%
export_bundle()   # zips results + analysis + banks; download from the Output tab