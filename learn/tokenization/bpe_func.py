#llm from scratch
# BPE Tokenization with functional approach

from collections import Counter

def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))
    
# count how many times a pair occured in the provided list of ids
def get_pair_counts(ids):
    return Counter(zip(ids, ids[1:]))

def merge_pair(ids, pair, new_id):
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

def train_bpe(ids, num_merges: int, min_count: int = 2):
    merges = {}
    for _ in range(num_merges):
        pair_counts = get_pair_counts(ids)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        if pair_counts[best_pair] < min_count:
            break
        new_id = 256 + len(merges)
        ids = merge_pair(ids, best_pair, new_id)
        merges = {**merges, best_pair: new_id}
    return merges

def build_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}

    for (a,b),new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

#encode new text with the learned merges
def encoder(text, merges):
    ids = encode(text)
    for pair, new_id in merges.items():
        ids = merge_pair(ids, pair, new_id)
    return ids

from functools import reduce
def encode_reduce(text, merges):
    start = list(text.encode("utf-8"))
    return reduce(lambda ids, item: merge_pair(ids, item[0], item[1]), merges.items(), start)

text = input("please enter train text: ")
num_merges = int(input("please provide num of merges: "))
sample = encode(text)
merges = train_bpe(sample, num_merges)
print(merges)
vocab = build_vocab(merges)
print(vocab[259])
print(len(vocab))
print("decoded text is: ")
print(decode([257, 259, 114, 259, 115, 116, 32, 99, 97, 102, 195, 169], vocab))
print(decode([195], vocab))