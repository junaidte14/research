import re
from collections import Counter, defaultdict

class BPETokenizer:

    PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")

    def __init__(self):
        self._reset()

    def _reset(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
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
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
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
            new_id = 256 + len(self.merges)
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
                self._cache[chunk] = self._encode_chunk(list(chunk.encode("utf-8")))
            ids.extend(self._cache[chunk])
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

""" tok = BPETokenizer()
tok.train("low lower lowest café", num_merges=5)

print(tok.merges)
print(tok.train_stats())

ids = tok.encode("lowest slow café")

print(ids)
print(tok.decode(ids))
print(tok.decode([195])) """