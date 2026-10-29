# %% [markdown]
# # LAR v9 — controlled re-run (resumable Kaggle notebook)
# 
# **What this notebook does.** Re-runs the Latent Attribution Regularization experiments under a stricter protocol, with seeds **0–9 in every phase**:
# 
# * new **controls/baselines**: augmentation-only (`aug_only`), prediction-consistency (`pred_consistency`)
# * new **LAR variants**: alignment-aware training loss (`lar_aligned`), correlation loss (`lar_pearson`), plus the original loss as in v8 (`lar_pos`)
# * **validated perturbations** (independent classifier + sentence-embedding filter, thresholds fixed in `BANK_PARAMS`), built once and cached, identical for every condition and seed
# * evaluation of every trained model in one pass: full-validation accuracy, stability (naive-positional vs greedy vs order-preserving alignment — a controlled version of the alignment-artifact finding), alignment coverage, prediction-flip rate, and faithfulness (comprehensiveness / sufficiency vs a random-ranking baseline)
# * paired statistics (bootstrap CI, d_z, t and Wilcoxon, Holm correction) and LaTeX tables
# 
# **One-time setup** (same as v8): Notebook Settings → Internet ON; Add-ons → Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`; accelerator GPU. Slug used: `lar-v9-results` (a *new* dataset, never mixed with v8). After the first commit, attach that dataset as an input once.
# 
# **Every session:** run cells 1–5 (definitions), then run whichever *phase cell* you need. Finished `(phase, condition, seed)` triples are skipped; a half-trained model resumes from its checkpoint; a changed config raises an error instead of silently mixing protocols.
# 
# **Order:** `S_smoke` (plumbing check, ~minutes) → `A` → **gate** → `B` → `C`/`D` → `E` → analysis. Phases A and B share SST-2/small, so contrasts are computed across both.

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
LOG_EVERY = 250
BASE_MODEL_NAME = "distilbert-base-uncased"

DATASETS = {
    "sst2": dict(hf=("nyu-mll/glue", "sst2"), text="sentence", train="train", eval="validation",
                 max_len=64, truncate_words=None, shuffle_train=False, eval_n=None),
    "rotten_tomatoes": dict(hf=("cornell-movie-review-data/rotten_tomatoes", None), text="text", train="train", eval="validation",
                            max_len=64, truncate_words=None, shuffle_train=True, eval_n=None),
    "imdb": dict(hf=("stanfordnlp/imdb", None), text="text", train="train", eval="test",
                 max_len=128, truncate_words=90, shuffle_train=True, eval_n=872),
}

SCALES = {
    "smoke": dict(train_size=120, epochs=1, ckpt_every=60, commit_ckpt_every=None, eval_limit=60),
    "small": dict(train_size=1500, epochs=1, ckpt_every=500, commit_ckpt_every=None, eval_limit=None),
    "large": dict(train_size=20000, epochs=2, ckpt_every=2000, commit_ckpt_every=6000, eval_limit=None),
}
CONDITIONS = {
    "ce_only_control":  dict(kind="ce"),
    "aug_only":         dict(kind="aug"),                                   # 0.5*(CE(x)+CE(x~))
    "lar_pos":          dict(kind="lar", align="positional", dist="cos", lam=0.5),   # LAR exactly as in v8
    "lar_aligned":      dict(kind="lar", align="seq", dist="cos", lam=0.5),          # alignment-aware training loss
    "lar_pearson":      dict(kind="lar", align="seq", dist="pearson", lam=0.5),      # FAR-style correlation loss
    "pred_consistency": dict(kind="kl", lam=0.5),                           # symmetric KL between predictions
}
PHASES = {
    "S_smoke":               dict(dataset="sst2", scale="smoke", conditions=list(CONDITIONS)),
    "A_sst2_small_core":     dict(dataset="sst2", scale="small",
                                  conditions=["ce_only_control", "aug_only", "lar_pos"]),
    "B_sst2_small_variants": dict(dataset="sst2", scale="small",
                                  conditions=["lar_aligned", "lar_pearson", "pred_consistency"]),
    "C_rt_small_edited":            dict(dataset="rotten_tomatoes", scale="small",
                                  conditions=["ce_only_control", "aug_only", "lar_pos", "lar_aligned"]),
    "D_imdb_small":          dict(dataset="imdb", scale="small",
                                  conditions=["ce_only_control", "aug_only", "lar_pos", "lar_aligned"]),
    "E_sst2_large":          dict(dataset="sst2", scale="large",
                                  conditions=["ce_only_control", "aug_only", "lar_pos", "lar_aligned"]),
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
                store["rows"].append(dict(phase=phase_name, dataset=p["dataset"], scale=p["scale"], condition=cond_name,
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


def load_df(include_smoke=False):
    df = pd.DataFrame(load_store()["rows"])
    if df.empty:
        return df
    if not include_smoke:
        df = df[~df.phase.str.startswith("S_")]
    # a (dataset, scale, condition, seed, strength) must be unique across phases; keep the first and warn
    key = ["dataset", "scale", "condition", "seed", "strength"]
    dup = df.duplicated(key, keep="first")
    if dup.any():
        print(f"WARNING: {int(dup.sum())} duplicate rows across phases dropped")
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
    """Paired-by-seed contrast of `treat` minus `ref` for one metric inside each group."""
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
        try:
            wp = sst.wilcoxon(d).pvalue
        except ValueError:
            wp = float("nan")
        g = g if isinstance(g, tuple) else (g,)
        rows.append(dict(zip(group_cols, g), metric=metric, treat=treat, ref=ref, n_seeds=len(d),
                         mean_diff=d.mean(), ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                         d_z=d.mean() / sd if sd > 0 else float("nan"), p_t=t.pvalue, p_wilcoxon=wp,
                         n_pos=int((d > 0).sum())))
    return pd.DataFrame(rows)


def run_analysis(verbose=True):
    df = load_df()
    if df.empty:
        print("no results yet"); return None
    out = {}
    strength_groups = ["dataset", "scale", "strength"]
    model_groups = ["dataset", "scale"]
    model_df = df.drop_duplicates(["dataset", "scale", "condition", "seed"])

    # 1) primary family: stability (rho_seq) of each treatment vs aug_only, Holm-corrected within dataset/scale/strength
    prim = []
    for tr in HOLM_FAMILY:
        prim.append(paired_contrast(df, "rho_seq", tr, REF_FOR_AUG, strength_groups))
    prim = pd.concat([p for p in prim if len(p)], ignore_index=True) if any(len(p) for p in prim) else pd.DataFrame()
    if len(prim):
        prim["p_holm"] = np.nan
        for _, idx in prim.groupby(strength_groups).groups.items():
            prim.loc[idx, "p_holm"] = holm(prim.loc[idx, "p_t"].values)
    out["primary_vs_aug"] = prim

    # 2) context contrasts (descriptive, not in the Holm family)
    ctx = pd.concat([paired_contrast(df, "rho_seq", "lar_pos", "ce_only_control", strength_groups),
                     paired_contrast(df, "rho_seq", "aug_only", "ce_only_control", strength_groups)], ignore_index=True)
    out["context"] = ctx

    # 3) accuracy / faithfulness / robustness side-effects (model-level metrics), every treatment vs aug_only
    side = []
    for m in ["acc_full", "comp_gain", "suff_gain"]:
        for tr in HOLM_FAMILY + ["ce_only_control"]:
            side.append(paired_contrast(model_df, m, tr, REF_FOR_AUG, model_groups))
    side.append(paired_contrast(df, "flip_rate", "lar_pos", REF_FOR_AUG, strength_groups))
    out["side_effects"] = pd.concat([s for s in side if len(s)], ignore_index=True) if any(len(s) for s in side) else pd.DataFrame()

    # 4) controlled alignment-artifact ablation: same models, three alignment procedures
    art = df.groupby(["dataset", "scale", "condition", "strength"])[["rho_positional", "rho_greedy", "rho_seq", "coverage"]].mean().reset_index()
    out["alignment_ablation"] = art

    # 5) descriptive table: mean (sd) over seeds
    desc = df.groupby(["dataset", "scale", "condition", "strength"]).rho_seq.agg(["mean", "std", "count"]).reset_index()
    out["descriptive"] = desc
    mdesc = model_df.groupby(["dataset", "scale", "condition"])[["acc_full", "comp_gain", "suff_gain"]].agg(["mean", "std"]).reset_index()
    out["model_descriptive"] = mdesc

    for k, v in out.items():
        if len(v):
            v.to_csv(os.path.join(ANALYSIS_DIR, f"{k}.csv"), index=False)
    if verbose:
        pd.set_option("display.width", 220, "display.max_columns", 30)
        for k in ["primary_vs_aug", "context", "alignment_ablation", "model_descriptive"]:
            print(f"\n----- {k} -----")
            print(out[k].round(4).to_string(index=False) if len(out[k]) else "(nothing yet)")
        print(f"\nCSV files written to {ANALYSIS_DIR}")
    return out


def gate_report():
    """Stage-0 decision: does LAR still beat the augmentation-only control? Read the CI, not just the p-value."""
    df = load_df()
    df = df[(df.dataset == "sst2") & (df.scale == "small")]
    if df.empty:
        print("phase A not run yet"); return
    for tr in ["lar_pos", "aug_only"]:
        ref = "aug_only" if tr == "lar_pos" else "ce_only_control"
        c = paired_contrast(df, "rho_seq", tr, ref, ["strength"])
        print(f"\n{tr} - {ref}  (rho_seq, paired by seed)")
        print(c[["strength", "n_seeds", "mean_diff", "ci_lo", "ci_hi", "d_z", "p_t", "p_wilcoxon", "n_pos"]].round(4).to_string(index=False))
    print("\nReading guide: LAR - aug_only with a 95% CI clearly above 0 => the attribution term adds something beyond "
          "data exposure (proceed). CI spanning 0 and aug_only - control clearly >0 => the gain is mostly augmentation "
          "(reframe the paper around the measurement fix + controlled study).")


def latex_main_table(dataset="sst2", scale="small", conditions=None, path=None):
    """Mean (sd) of rho_seq per condition x strength, plus Holm-adjusted p vs aug_only, as a booktabs table."""
    df = load_df()
    df = df[(df.dataset == dataset) & (df.scale == scale)]
    conditions = conditions or [c for c in CONDITIONS if c in set(df.condition)]
    prim = run_analysis(verbose=False)["primary_vs_aug"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Condition & Light & Medium & Heavy \\", r"\midrule"]
    for c in conditions:
        cells = []
        for s in EVAL_STRENGTHS:
            v = df[(df.condition == c) & (df.strength == s)].rho_seq
            cell = f"{v.mean():.4f} ({v.std():.4f})" if len(v) else "--"
            if len(prim):
                m = prim[(prim.dataset == dataset) & (prim.scale == scale) & (prim.strength == s) & (prim.treat == c)]
                if len(m):
                    cell += r"$^{\dagger}$" if m.p_holm.iloc[0] < 0.05 else ""
            cells.append(cell)
        name = c.replace("_", "\\_")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% $\dagger$: Holm-adjusted paired-$t$ p<0.05 versus aug\_only"]
    tex = "\n".join(lines)
    path = path or os.path.join(ANALYSIS_DIR, f"table_{dataset}_{scale}.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(tex)
    return tex


def export_bundle():
    """Zip results + analysis + banks into one file (download from the Output tab or attach to the paper repo)."""
    base = os.path.join(WORK, "lar_v9_export")
    shutil.make_archive(base, "zip", RUN_DIR, ".")
    print("wrote", base + ".zip")

# %% [markdown]
# ## Phase S — smoke test (run this first, once)
# Tiny run of all six conditions on one seed to confirm paths, banks, checkpoints, evaluation and auto-commit. Note the `s/step` printed per condition on your GPU and send it back so the full budget can be planned. Then `reset_phase("S_smoke")` so it never touches real results (analysis already ignores `S_` phases).

# %%
run_phase("S_smoke", seeds=[0])   # plumbing check only; analysis needs >=3 seeds

# %%
# reset_phase("S_smoke")   # uncomment after you have looked at the smoke output

# %% [markdown]
# ## Phase A — SST-2, small scale: control / augmentation-only / LAR (as in v8)  → the decisive gate
# 1,500 training examples, 1 epoch, seeds 0–9. After it finishes, `gate_report()` answers: *does LAR still beat augmentation-only?*

# %%
run_phase("A_sst2_small_core")

# %%
gate_report()

# %% [markdown]
# ## Phase B — SST-2, small scale: alignment-aware LAR, correlation-loss LAR, prediction-consistency baseline
# Run after the gate (or earlier if you prefer; it shares banks/seeds with phase A).

# %%
run_phase("B_sst2_small_variants")

# %% [markdown]
# ## Phase C — Rotten Tomatoes, small scale (second dataset)

# %%
run_phase("C_rt_small")

# %% [markdown]
# ## Phase D — IMDB (long reviews, truncated to 90 words / 128 tokens), small scale (third dataset)
# Sequences are ~2× longer than SST-2, so each LAR step is slower; check `s/step` in the log.

# %%
run_phase("D_imdb_small")

# %% [markdown]
# ## Phase E — SST-2, large scale (20,000 × 2 epochs) under the new protocol
# Heaviest phase. Edit `PHASES['E_sst2_large']['conditions']` in cell 1 **before the first run** to keep only the conditions you need after seeing phases A–B (e.g. drop `lar_pos` or `lar_aligned`). Checkpoints are committed to the dataset every 6,000 steps so a stopped session resumes mid-seed.

# %%
run_phase("E_sst2_large")

# %% [markdown]
# ## Analysis, tables, export
# Re-runnable at any time on whatever seeds have finished.

# %%
out = run_analysis()

# %%
latex_main_table("sst2", "small")
# latex_main_table("rotten_tomatoes", "small"); latex_main_table("imdb", "small"); latex_main_table("sst2", "large")

# %%
export_bundle()   # zips results + analysis + banks; download from the Output tab

