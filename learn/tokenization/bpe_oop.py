#learning llms from scratch
#3. BPE Tokenization (OOP Approach)

from collections import Counter

class BPETokenizer:

    def __init__(self):
        self._reset()

    def _reset(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.train_bytes = 0
        self.train_tokens = 0

    @staticmethod
    def _get_pair_counts(ids):
        return Counter(zip(ids, ids[1:]))

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

    def train(self, text, num_merges, min_count=2):
        self._reset()
        ids = list(text.encode("utf-8"))
        self.train_bytes = len(ids)
        for _ in range(num_merges):
            pair_counts = self._get_pair_counts(ids)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_count:
                break
            new_id = 256 + len(self.merges)
            ids = self._merge_pair(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
        self.train_tokens = len(ids)
            
    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            ids = self._merge_pair(ids, pair, new_id)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

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

""" print("#" * 45)
print("BPE Tokenizer is Ready!")
print("#" * 45)
train_text = input("please provide train text -> ")
num_merges = int(input("please provide number of merges -> "))

tok = BPETokenizer()
tok.train(train_text, num_merges)
print("#" * 45)
print("Tokenization is done and following are the stats:")
print("#" * 45)
for key, value in tok.stats(train_text).items():
    print(f"{key}: {value}") """