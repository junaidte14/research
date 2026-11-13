# -*- coding: utf-8 -*-
"""
LAR Kaggle Full-Scale Pipeline (v8-Kaggle, self-persisting)
====================================================================

  1. RESTORE ON STARTUP: if you attach a previous run's output as an input
     Dataset, this script copies results.json/csv and any checkpoints from
     it into /kaggle/working before doing anything else.

  2. AUTO-COMMIT DURING THE RUN: using the Kaggle API (via Kaggle Secrets
     for credentials), this script periodically pushes /kaggle/working's
     contents as a new version of a Kaggle Dataset -- automatically, no
     manual re-upload needed. It commits after every completed seed, and
     every AUTO_COMMIT_EVERY_N_CHECKPOINTS local checkpoints during a long
     seed's training, so a crash loses at most a partial checkpoint
     interval, not a whole seed.

  Auto-commit is best-effort and NEVER blocks or crashes training: if
  Secrets aren't configured, internet is off, or the API call fails for
  any reason, it prints a warning and continues. Local saves to
  /kaggle/working always happen regardless.

ONE-TIME SETUP (do this before the first run):
  A) Notebook Settings > Internet: ON (required for the Kaggle API).
  B) Add-ons > Secrets: add KAGGLE_USERNAME and KAGGLE_KEY (your Kaggle
     API token -- generate one at kaggle.com/settings > API > Create New
     Token, which gives you a kaggle.json with these two values).
  C) Set KAGGLE_DATASET_SLUG below to a dataset you own (it will be
     created automatically on first commit if it doesn't exist yet).
  D) After your FIRST run (even a partial one), go to kaggle.com/datasets,
     find the dataset this script created, and attach it as an input to
     this notebook (Add Input > search for it). From then on, every
     session automatically restores from it AND commits back to it --
     fully closing the loop.

HOW TO RUN ACROSS MULTIPLE KAGGLE SESSIONS:
  1. First run: just run it (auto-commit creates the dataset for you).
  2. Attach that dataset as input (step D above) -- only needs doing once.
  3. If the session stops for any reason, reopen and "Run All" again.
     It restores from the input dataset, skips finished seeds, and
     resumes the in-progress seed from its last auto-committed checkpoint.
"""

import glob
import json
import os
import random
import shutil
import subprocess

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.stats import spearmanr
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# 0. Kaggle persistence config -- EDIT THESE
# ============================================================
KAGGLE_DATASET_SLUG = "lar-paper-checkpoints"      # <<< a dataset slug you own
KAGGLE_DATASET_TITLE = "LAR Paper Checkpoints"     # <<< human-readable title
INPUT_DATASET_DIR = f"/kaggle/input/{KAGGLE_DATASET_SLUG}"  # where it appears once attached as input
AUTO_COMMIT_TO_KAGGLE = True    # set False to disable API auto-commit entirely
AUTO_COMMIT_EVERY_N_CHECKPOINTS = 3  # commit every Nth local checkpoint (reduces upload overhead)

OUTPUT_DIR = "/kaggle/working/lar_paper_results"
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(OUTPUT_DIR, "lar_v8_results.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "lar_v8_results.csv")

for pkg in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
            "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.corpus import wordnet as wn  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

BASE_MODEL_NAME = "distilbert-base-uncased"


# ============================================================
# 1. Restore from a previously-attached input dataset (if any)
# ============================================================
def bootstrap_from_input_dataset():
    if not os.path.isdir(INPUT_DATASET_DIR):
        print(f"No input dataset found at {INPUT_DATASET_DIR} -- starting fresh "
              f"(this is expected on your very first run).")
        return

    for fname in ("lar_v8_results.json", "lar_v8_results.csv"):
        src = os.path.join(INPUT_DATASET_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Restored {fname} from input dataset.")

    src_ckpt_dir = os.path.join(INPUT_DATASET_DIR, "checkpoints")
    if os.path.isdir(src_ckpt_dir):
        for src_path in glob.glob(os.path.join(src_ckpt_dir, "*.pt")):
            fname = os.path.basename(src_path)
            dst_path = os.path.join(CKPT_DIR, fname)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                print(f"Restored checkpoint {fname} from input dataset.")

    print("Bootstrap from input dataset complete.")


bootstrap_from_input_dataset()

# ============================================================
# 2. Kaggle API auto-commit (best-effort, never blocks training)
# ============================================================
_kaggle_api_ready = False


def _setup_kaggle_credentials():
    global _kaggle_api_ready
    if _kaggle_api_ready:
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        username = secrets.get_secret("KAGGLE_USERNAME")
        key = secrets.get_secret("KAGGLE_KEY")
        cred_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(cred_dir, exist_ok=True)
        cred_path = os.path.join(cred_dir, "kaggle.json")
        with open(cred_path, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(cred_path, 0o600)
        _kaggle_api_ready = True
        return True
    except Exception as e:
        print(f"  [auto-commit] Kaggle Secrets not configured ({e}); "
              f"auto-commit disabled for this session. Local saves still work.")
        return False


def kaggle_auto_commit(message):
    """Push OUTPUT_DIR as a new version of the Kaggle Dataset. Best-effort:
    any failure prints a warning and returns, never raises."""
    if not AUTO_COMMIT_TO_KAGGLE:
        return
    if not _setup_kaggle_credentials():
        return
    try:
        from kaggle_secrets import UserSecretsClient
        username = UserSecretsClient().get_secret("KAGGLE_USERNAME")

        meta = {
            "title": KAGGLE_DATASET_TITLE,
            "id": f"{username}/{KAGGLE_DATASET_SLUG}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(os.path.join(OUTPUT_DIR, "dataset-metadata.json"), "w") as f:
            json.dump(meta, f)

        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", OUTPUT_DIR, "-m", message],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"  [auto-commit] Pushed new dataset version: {message}")
            return

        # Likely the dataset doesn't exist yet -- create it once.
        create_result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", OUTPUT_DIR],
            capture_output=True, text=True, timeout=600
        )
        if create_result.returncode == 0:
            print(f"  [auto-commit] Created dataset {KAGGLE_DATASET_SLUG} (first commit).")
        else:
            print(f"  [auto-commit] WARNING: version failed ({result.stderr[:200]}); "
                  f"create also failed ({create_result.stderr[:200]})")
    except Exception as e:
        print(f"  [auto-commit] WARNING: skipped due to error: {e}")


# ============================================================
# 3. Config
# ============================================================
SMOKE_TEST = False  # set True once to verify paths + auto-commit, then False

if SMOKE_TEST:
    TRAIN_SIZE = 300
    EPOCHS = 1
    SEEDS = [0]
    EVAL_SIZE = 50
    CHECKPOINT_EVERY = 100
    print(">>> SMOKE_TEST=True -- tiny check, including a test auto-commit. Set False for real run. <<<")
else:
    TRAIN_SIZE = 20000
    EPOCHS = 2
    SEEDS = [5, 6, 7, 8, 9]  # <<< extending from 5->10 total seeds to regain statistical power;
                              #     seeds 0-4 are already complete and will be skipped automatically
                              #     once lar_v8_results.json (already in your dataset) is restored
    EVAL_SIZE = 300
    CHECKPOINT_EVERY = 2000

LEARNING_RATE = 3e-5
MAX_LEN = 64
LOG_EVERY = 500

TRAIN_PERTURBATION_STRENGTH = 0.35
EVAL_PERTURBATION_STRENGTHS = {"light": 0.15, "medium": 0.35, "heavy": 0.60}

CONDITIONS = {
    "ce_only_control": 0.0,
    "lar_lambda_0.5": 0.5,
}

# ============================================================
# 4. WordNet synonym perturbation
# ============================================================
def _wordnet_pos(nltk_tag):
    if nltk_tag.startswith("J"):
        return wn.ADJ
    if nltk_tag.startswith("R"):
        return wn.ADV
    return None


def _find_candidates(text):
    words = text.split()
    if not words:
        return words, []
    tagged = nltk.pos_tag(words)
    candidates = []
    for i, (w, tag) in enumerate(tagged):
        wpos = _wordnet_pos(tag)
        if wpos is None:
            continue
        clean = w.strip(".,!?;:\"'").lower()
        if not clean:
            continue
        synsets = wn.synsets(clean, pos=wpos)
        lemmas = set()
        for s in synsets:
            for lemma in s.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != clean and " " not in name:
                    lemmas.add(name)
        if lemmas:
            candidates.append((i, w, list(lemmas)))
    return words, candidates


def wordnet_synonym_swap(text, rng, strength=0.35):
    words, candidates = _find_candidates(text)
    if not candidates:
        return text, False
    n_swaps = max(1, round(strength * len(candidates)))
    rng.shuffle(candidates)
    chosen = candidates[:n_swaps]
    new_words = words[:]
    for i, orig_w, lemmas in chosen:
        repl = rng.choice(lemmas)
        trailing = ""
        core = orig_w
        while core and core[-1] in ".,!?;:":
            trailing = core[-1] + trailing
            core = core[:-1]
        if core.istitle():
            repl = repl.capitalize()
        new_words[i] = repl + trailing
    perturbed = " ".join(new_words)
    return perturbed, perturbed != text


# ============================================================
# 5. Model / attribution utilities
# ============================================================
def load_fresh_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2).to(device)
    return model, tokenizer


def compute_attribution_vector(model, tokenizer, text_input, target_class):
    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    embedding_layer = model.distilbert.embeddings.word_embeddings
    embeds = embedding_layer(inputs["input_ids"])
    embeds.requires_grad_(True)
    outputs = model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"])
    target_logit = outputs.logits[0, target_class]
    grads = torch.autograd.grad(target_logit, embeds, retain_graph=False, create_graph=False)[0]
    attr = torch.norm(grads, dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, attr.detach().cpu().numpy()


def evaluate_aligned_stability(model, tokenizer, dataset, perturb_fn, rng, num_samples):
    correlations = []
    attempted = 0
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        orig_text = sample["sentence"]
        label = sample["label"]
        pert_text, changed = perturb_fn(orig_text, rng)
        if not changed:
            continue
        attempted += 1
        try:
            tokens_orig, attr_orig = compute_attribution_vector(model, tokenizer, orig_text, label)
            tokens_pert, attr_pert = compute_attribution_vector(model, tokenizer, pert_text, label)
            matching_orig, matching_pert = [], []
            used_pert_idx = set()
            for oi, otok in enumerate(tokens_orig):
                if otok in ("[CLS]", "[SEP]"):
                    continue
                for pi, ptok in enumerate(tokens_pert):
                    if pi in used_pert_idx:
                        continue
                    if ptok == otok:
                        matching_orig.append(oi)
                        matching_pert.append(pi)
                        used_pert_idx.add(pi)
                        break
            if len(matching_orig) >= 3:
                s_o = attr_orig[matching_orig]
                s_p = attr_pert[matching_pert]
                corr, _ = spearmanr(s_o, s_p)
                if not np.isnan(corr):
                    correlations.append(corr)
        except Exception:
            continue
    if not correlations:
        return float("nan"), 0, attempted
    return float(np.mean(correlations)), len(correlations), attempted


def evaluate_accuracy(model, tokenizer, dataset, num_samples):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            inputs = tokenizer(sample["sentence"], return_tensors="pt",
                               truncation=True, max_length=MAX_LEN).to(device)
            pred = model(**inputs).logits.argmax(dim=-1).item()
            correct += int(pred == sample["label"])
            total += 1
    return correct / total if total else float("nan")


# ============================================================
# 6. LAR loss (double-backward)
# ============================================================
def compute_lar_loss(model, input_ids_orig, input_ids_pert, target_labels, lambda_reg):
    embedding_layer = model.distilbert.embeddings.word_embeddings

    if lambda_reg == 0.0:
        embeds_orig = embedding_layer(input_ids_orig)
        outputs_orig = model(inputs_embeds=embeds_orig)
        ce_loss = nn.CrossEntropyLoss()(outputs_orig.logits, target_labels)
        return ce_loss, ce_loss.item(), 0.0

    # DISABLE Flash/SDPA kernels during LAR calculation to allow double-backward
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        embeds_orig = embedding_layer(input_ids_orig)
        outputs_orig = model(inputs_embeds=embeds_orig)
        logits_orig = outputs_orig.logits
        ce_loss = nn.CrossEntropyLoss()(logits_orig, target_labels)

        target_logits = logits_orig.gather(1, target_labels.unsqueeze(-1)).squeeze(-1)
        grads_orig = torch.autograd.grad(
            outputs=target_logits.sum(), inputs=embeds_orig,
            create_graph=True, retain_graph=True
        )[0]
        attr_orig = torch.norm(grads_orig, dim=-1)
        attr_orig_norm = attr_orig / (torch.norm(attr_orig, dim=-1, keepdim=True) + 1e-8)

        embeds_pert = embedding_layer(input_ids_pert)
        outputs_pert = model(inputs_embeds=embeds_pert)
        logits_pert = outputs_pert.logits
        target_logits_pert = logits_pert.gather(1, target_labels.unsqueeze(-1)).squeeze(-1)
        grads_pert = torch.autograd.grad(
            outputs=target_logits_pert.sum(), inputs=embeds_pert,
            create_graph=True, retain_graph=True
        )[0]
        attr_pert = torch.norm(grads_pert, dim=-1)
        attr_pert_norm = attr_pert / (torch.norm(attr_pert, dim=-1, keepdim=True) + 1e-8)

        min_len = min(attr_orig_norm.shape[1], attr_pert_norm.shape[1])
        cos_sim = torch.cosine_similarity(
            attr_orig_norm[:, :min_len], attr_pert_norm[:, :min_len], dim=-1
        )
        lar_loss = torch.mean(1.0 - cos_sim)
        total_loss = ce_loss + (lambda_reg * lar_loss)
        return total_loss, ce_loss.item(), lar_loss.item()

# ============================================================
# 7. Deterministic multi-epoch step sequence
# ============================================================
def build_step_sequence(dataset_len, train_size, epochs, seed):
    rng = random.Random(seed)
    n = min(train_size, dataset_len)
    sequence = []
    for _ in range(epochs):
        idxs = list(range(n))
        rng.shuffle(idxs)
        sequence.extend(idxs)
    return sequence


# ============================================================
# 8. Checkpointing
# ============================================================
def checkpoint_path(condition, seed):
    return os.path.join(CKPT_DIR, f"{condition}_seed{seed}.pt")


def save_checkpoint(condition, seed, global_step, model, optimizer, perturb_rng):
    torch.save({
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "perturb_rng_state": perturb_rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, checkpoint_path(condition, seed))


def load_checkpoint_if_exists(condition, seed, model, optimizer, perturb_rng):
    path = checkpoint_path(condition, seed)
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    perturb_rng.setstate(ckpt["perturb_rng_state"])
    trs = ckpt["torch_rng_state"]
    torch.set_rng_state(trs.cpu() if torch.is_tensor(trs) else trs)
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    print(f"  Resumed {condition} seed={seed} from checkpoint at step {ckpt['global_step']}")
    return ckpt["global_step"]


def delete_checkpoint(condition, seed):
    path = checkpoint_path(condition, seed)
    if os.path.exists(path):
        os.remove(path)


# ============================================================
# 9. Training loop -- auto-commits during long seeds, not just after
# ============================================================
def train_condition(condition, lambda_reg, train_dataset, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, tokenizer = load_fresh_model()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    perturb_rng = random.Random(seed)

    start_step = load_checkpoint_if_exists(condition, seed, model, optimizer, perturb_rng)

    sequence = build_step_sequence(len(train_dataset), TRAIN_SIZE, EPOCHS, seed)
    total_steps = len(sequence)

    if start_step >= total_steps:
        print(f"  {condition} seed={seed} already fully trained ({total_steps} steps). Skipping training.")
        model.eval()
        return model, tokenizer

    model.train()
    running_correct, running_total = 0, 0
    checkpoint_count = 0

    for global_step in range(start_step, total_steps):
        idx = sequence[global_step]
        sample = train_dataset[idx]
        text = sample["sentence"]
        label_val = sample["label"]
        label = torch.tensor([label_val]).to(device)

        pert_text, changed = wordnet_synonym_swap(text, perturb_rng, strength=TRAIN_PERTURBATION_STRENGTH)
        if not changed:
            pert_text = text

        inputs_orig = tokenizer(text, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)
        inputs_pert = tokenizer(pert_text, return_tensors="pt", max_length=MAX_LEN, truncation=True).to(device)

        optimizer.zero_grad()
        loss, ce_val, lar_val = compute_lar_loss(
            model, inputs_orig["input_ids"], inputs_pert["input_ids"], label, lambda_reg
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = model(**inputs_orig).logits.argmax(dim=-1).item()
            running_correct += int(pred == label_val)
            running_total += 1

        step_num = global_step + 1
        if step_num % LOG_EVERY == 0:
            running_acc = running_correct / running_total
            print(f"  [{condition}, seed={seed}] step {step_num}/{total_steps} "
                  f"| total={loss.item():.4f} ce={ce_val:.4f} lar={lar_val:.4f} "
                  f"| running_train_acc={running_acc:.3f}")
            running_correct, running_total = 0, 0

        if step_num % CHECKPOINT_EVERY == 0:
            save_checkpoint(condition, seed, step_num, model, optimizer, perturb_rng)
            checkpoint_count += 1
            print(f"    [local checkpoint saved at step {step_num}]")
            if checkpoint_count % AUTO_COMMIT_EVERY_N_CHECKPOINTS == 0:
                kaggle_auto_commit(f"{condition} seed={seed} step={step_num}/{total_steps}")

    save_checkpoint(condition, seed, total_steps, model, optimizer, perturb_rng)
    model.eval()
    return model, tokenizer


# ============================================================
# 10. Results save/load
# ============================================================
def load_existing_results():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
        print(f"Found existing results: {len(results)} rows.")
        return results
    return []


def is_seed_complete(results, condition, seed):
    strengths_present = {
        r["perturbation_strength"] for r in results
        if r["condition"] == condition and r["seed"] == seed
    }
    return strengths_present == set(EVAL_PERTURBATION_STRENGTHS.keys())


def save_partial(results):
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)


# ============================================================
# 11. Main
# ============================================================
def main():
    print("Loading SST-2...")
    train_dataset = load_dataset("nyu-mll/glue", "sst2", split="train")
    val_dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")

    results = load_existing_results()
    total_runs = len(CONDITIONS) * len(SEEDS)
    run_i = 0

    try:
        for cond_name, lambda_reg in CONDITIONS.items():
            for seed in SEEDS:
                run_i += 1
                if is_seed_complete(results, cond_name, seed):
                    print(f"\n=== [{run_i}/{total_runs}] {cond_name} seed={seed}: already complete, skipping ===")
                    continue

                print(f"\n=== [{run_i}/{total_runs}] {cond_name} (lambda={lambda_reg}) | seed {seed} "
                      f"| TRAIN_SIZE={TRAIN_SIZE} x {EPOCHS} epochs ===")
                model, tokenizer = train_condition(cond_name, lambda_reg, train_dataset, seed)

                acc = evaluate_accuracy(model, tokenizer, val_dataset, num_samples=EVAL_SIZE)
                print(f"  val_accuracy={acc:.4f}")

                for strength_name, frac in EVAL_PERTURBATION_STRENGTHS.items():
                    eval_rng = random.Random(1000 + seed)

                    def perturb_fn(text, rng, _frac=frac):
                        return wordnet_synonym_swap(text, rng, strength=_frac)

                    mean_rho, n_valid, n_attempted = evaluate_aligned_stability(
                        model, tokenizer, val_dataset, perturb_fn, eval_rng, num_samples=EVAL_SIZE
                    )
                    print(f"    [{strength_name} perturbation, frac={frac}] "
                          f"mean_rho={mean_rho:.4f} (n={n_valid}/{n_attempted})")

                    results.append({
                        "condition": cond_name,
                        "lambda": lambda_reg,
                        "seed": seed,
                        "train_size": TRAIN_SIZE,
                        "epochs": EPOCHS,
                        "smoke_test": SMOKE_TEST,
                        "perturbation_strength": strength_name,
                        "perturbation_fraction": frac,
                        "mean_rho": mean_rho,
                        "n_valid_pairs": n_valid,
                        "n_attempted": n_attempted,
                        "val_accuracy": acc,
                    })

                save_partial(results)
                print(f"  Saved locally: {RESULTS_JSON} ({len(results)} rows total)")
                delete_checkpoint(cond_name, seed)

                # Always commit at a seed boundary -- this is the safest point.
                kaggle_auto_commit(f"completed {cond_name} seed={seed}")

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except (KeyboardInterrupt, Exception) as e:
        # Best-effort final save + commit even on crash/manual stop, then re-raise.
        print(f"\n!!! Interrupted or errored ({e}) -- attempting final save + commit before exiting !!!")
        save_partial(results)
        kaggle_auto_commit("interrupted -- emergency checkpoint commit")
        raise

    df = pd.DataFrame(results)
    if len(df):
        summary = df.groupby(["condition", "lambda", "perturbation_strength"]).agg(
            mean_rho_avg=("mean_rho", "mean"),
            mean_rho_std=("mean_rho", "std"),
            acc_avg=("val_accuracy", "mean"),
            acc_std=("val_accuracy", "std"),
        ).reset_index()
        print("\n" + "=" * 90)
        print(f"SUMMARY (SMOKE_TEST={SMOKE_TEST})")
        print("=" * 90)
        print(summary.to_string(index=False))

    all_done = all(
        is_seed_complete(results, cond, seed)
        for cond in CONDITIONS for seed in SEEDS
    )
    print("\n" + "=" * 70)
    if all_done:
        print("ALL SEEDS COMPLETE.")
        kaggle_auto_commit("all seeds complete -- final results")
        print(f"Final results also available locally at: {RESULTS_JSON}")
    else:
        print("Session ended with some seeds still incomplete (expected at this scale).")
        print("If auto-commit is working, just reopen and Run All -- it restores from")
        print("the input dataset automatically. If not yet attached as input, attach")
        print(f"'{KAGGLE_DATASET_SLUG}' as input first (see setup notes at the top).")
    print("=" * 70)

    return df


if __name__ == "__main__":
    df = main()