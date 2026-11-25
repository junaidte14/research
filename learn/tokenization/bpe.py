import re, json
from collections import Counter, defaultdict

class BPETokenizer:

    PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")

    def __init__(self, special_tokens=None):
        self._set_special_tokens(special_tokens or [])
        self._reset()

    def _set_special_tokens(self, special_tokens):
        self.special_tokens = list(special_tokens)
        self.n_special = len(self.special_tokens)
        self.special_ids = {token: i for i, token in enumerate(self.special_tokens)}

    def _reset(self):
        self.merges = {}
        self.vocab = {i: token.encode("utf-8") for i, token in enumerate(self.special_tokens)}
        for b in range(256):
            self.vocab[self.n_special + b] = bytes([b])
        self.train_bytes = 0
        self.train_tokens = 0
        self._cache = {}

    @staticmethod
    def _merge_pair(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and pair == (ids[i], ids[i+1]):
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _count_words(self, text):
        chunks = self.PATTERN.findall(text)
        chunk_counts = Counter(chunks)
        words = [[b + self.n_special for b in chunk.encode("utf-8")] for chunk in chunk_counts]
        freqs = list(chunk_counts.values())
        return words, freqs

    @staticmethod
    def _count_pairs(words, freqs):
        pair_counts = Counter()
        pair_to_words = defaultdict(set)

        for idx, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freqs[idx]
                pair_to_words[pair].add(idx)
        return pair_counts, pair_to_words

    def _apply_merge(self, words, freqs, pair_counts, pair_to_words, best_pair, new_id):
        for idx in list(pair_to_words[best_pair]):
            old_word = words[idx]
            
            for pair in zip(old_word, old_word[1:]):
                pair_counts[pair] -= freqs[idx]

            new_word = self._merge_pair(old_word, best_pair, new_id)
            words[idx] = new_word

            for pair in zip(new_word, new_word[1:]):
                pair_counts[pair] += freqs[idx]
                pair_to_words[pair].add(idx)

        for pair in [p for p, c in pair_counts.items() if c <= 0]:
            del pair_counts[pair]

    def train(self, text, num_merges, min_count = 2):
        self._reset()
        words, freqs = self._count_words(text)
        #print("".join(self.decode(w) for w in words))
        #print(words)
        #print(freqs)
        self.train_bytes = sum(len(w) * f for w, f in zip(words, freqs))
        pair_counts, pair_to_words = self._count_pairs(words, freqs)
        #print("")
        #print(pair_counts)
        #print(pair_to_words)
        #num_merges = 2
        for _ in range(num_merges):
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            #print("")
            #print(best_pair)
            if pair_counts[best_pair] < min_count:
                break
            new_id = len(self.vocab)
            #print(new_id)
            self.merges[best_pair] = new_id
            #print(self.merges)
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            #print(self.vocab[new_id])
            self._apply_merge(words, freqs, pair_counts, pair_to_words, best_pair, new_id)

        self.train_tokens = sum(len(w) * f for w, f in zip(words, freqs))

    def _encode_chunk(self, ids):
        while (len(ids) >= 2):
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = self._merge_pair(ids, best, self.merges[best])
        return ids

    def encode(self, text):
        ids = []
        for chunk in self.PATTERN.findall(text):
            if chunk not in self._cache:
                start_ids = [b + self.n_special for b in chunk.encode("utf-8")]
                self._cache[chunk] = self._encode_chunk(start_ids)
            ids.extend(self._cache[chunk])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "pattern": self.PATTERN.pattern,
            "special_tokens": self.special_tokens,
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        assert data["pattern"] == self.PATTERN.pattern, "pattern mismatch"
        self._set_special_tokens(data["special_tokens"])
        self._reset()
        for a, b, new_id in data["merges"]:
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def train_stats(self):
        if not self.train_tokens:
            return {"training_bytes": 0, "training_tokens": 0, "compression_ratio (bytes/token)": 0}
        else:
            return {
                "training_bytes": self.train_bytes, 
                "training_tokens": self.train_tokens, 
                "compression_ratio (bytes/token)": round(self.train_bytes/self.train_tokens, 3)
            }
            
    def stats(self, text):
        ids = self.encode(text)
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(ids)
        n_words = len(text.split())
        used = set(ids)
        top = Counter(ids).most_common(3)
        return {
            "characters": n_chars,
            "bytes": n_bytes,
            "tokens": n_tokens,
            "compression_ratio (bytes/token)": round(n_bytes / n_tokens, 3) if n_tokens else 0.0,
            "chars_per_token": round(n_chars / n_tokens, 3) if n_tokens else 0.0,
            "tokens_per_word": round(n_tokens / n_words, 3) if n_words else 0.0,
            "vocab_size": len(self.vocab),
            "merges_learned": len(self.merges),
            "unique_tokens_used": len(used),
            "vocab_used_pct": round(100 * len(used) / len(self.vocab), 2),
            "top_tokens": [(self.vocab[t], c) for t, c in top],
        }

if __name__ == "__main__":
    
    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    print(len(tok.vocab), tok.special_ids, tok.vocab[0], tok.vocab[1], tok.vocab[256])

    tok.train("low lower lowest café", num_merges=5)
    
    print(tok.merges)
    
    ids = tok.encode("lowest slow café")
    
    print(ids)
    print(tok.decode(ids))
    print(tok.decode([0]))

    #train, and save
    import os, time
    import numpy as np
    from datasets import load_dataset

    out_dir = "data_own"
    os.makedirs(out_dir, exist_ok=True)

    train_ds = load_dataset("roneneldan/TinyStories", split="train")
    val_ds = load_dataset("roneneldan/TinyStories", split="validation")
    train_text = "\n".join(train_ds[:20000]["text"])

    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    tok.train(train_text, num_merges=8192 - len(tok.vocab))
    tok.save(f"{out_dir}/tokenizer_own.json")
    eos_id = tok.special_ids["<|endoftext|>"]
    print(len(tok.merges), eos_id, len(tok.vocab))


    #tokenize both splits
    counts = {}
    for name, ds in [("val", val_ds), ("train", train_ds)]:
        total_tokens = 0
        start = time.time()
        with open(f"{out_dir}/{name}.bin", "wb") as f:
            for batch_start in range(0, len(ds), 1000):
                texts = ds[batch_start:batch_start + 1000]["text"]
                chunk = []
                for text in texts:
                    chunk.extend(tok.encode(text))
                    chunk.append(eos_id)
                np.array(chunk, dtype=np.uint16).tofile(f)
                total_tokens += len(chunk)
                if (batch_start // 1000 + 1) % 100 == 0:
                    print(f"{name}: {batch_start + 1000:,} stories, {total_tokens:,} tokens")
        counts[name] = (len(ds), total_tokens)
        print(f"{name}: {len(ds):,} stories, {total_tokens:,} tokens, {time.time() - start:.0f}s")

    #verify, write meta.json, list the files
    for name, ds in [("val", val_ds), ("train", train_ds)]:
        data = np.memmap(f"{out_dir}/{name}.bin", dtype=np.uint16, mode="r")
        n_eos = int((data == eos_id).sum())
        print(name, len(data), n_eos, int(data.max()))
        assert len(data) == counts[name][1], f"{name}: token count mismatch"
        assert n_eos == counts[name][0], f"{name}: end-marker count mismatch"
        first_end = int(np.argmax(data == eos_id))
        print("  first story round trip:", tok.decode(data[:first_end].tolist()) == ds[0]["text"])

    meta = {
        "tokenizer": "own",
        "vocab_size": len(tok.vocab),
        "eos_token_id": eos_id,
        "dtype": "uint16",
        "train_stories": counts["train"][0],
        "train_tokens": counts["train"][1],
        "val_stories": counts["val"][0],
        "val_tokens": counts["val"][1],
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    for fname in sorted(os.listdir(out_dir)):
        print(fname, round(os.path.getsize(f"{out_dir}/{fname}") / 1e6, 1), "MB")

