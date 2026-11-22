from datasets import load_dataset
import time
from tokenizers import Tokenizer, models, trainers, decoders, pre_tokenizers
from bpe import BPETokenizer

class MyTokenizer:
    def __init__(self, ds_size, tok_type):
        if ds_size not in ("small", "full"):
            raise ValueError("ds_size must be 'small' or 'full'")
        if tok_type not in ("own", "lib"):
            raise ValueError("tok_type must be 'own' or 'lib'")
        self.ds_size = ds_size
        self.tok_type = tok_type
        self.train_stories = []
        self.heldout_stories = []
        self.train_text = ""
        self.heldout_text = ""

    @staticmethod
    def _read_stories(split, count):
        stories = []
        stream = load_dataset("roneneldan/TinyStories", split=split)
        for example in stream:
            stories.append(example["text"])
            if len(stories) == count:
                break
        return stories

    def load_ds(self):
        if self.ds_size == "small":
            stories = self._read_stories("validation", 350)
            self.train_stories = stories[:300]
            self.heldout_stories = stories[300:]
        else:
            self.train_stories = self._read_stories("train", 20000)
            self.heldout_stories = self._read_stories("validation", 1000)
        self.train_text = "\n".join(self.train_stories)
        self.heldout_text = "\n".join(self.heldout_stories)
        print(f"training stories: {len(self.train_stories)} ({len(self.train_text):,} chars)")
        print(f"held-out stories: {len(self.heldout_stories)} ({len(self.heldout_text):,} chars)")

    def train(self):
        if self.tok_type == "own":
            self._train_own()
        else:
            self._train_lib()

    def _train_own(self):
        tok = BPETokenizer()
        num_merges = 300 if self.ds_size == "small" else 7935
        print("training our own tokenizer ...")
        start = time.time()
        tok.train(self.train_text, num_merges=num_merges)
        print(f"training took: {time.time() - start:.1f} seconds")
        print(tok.train_stats())
        print(tok.stats(self.heldout_text))

    def _train_lib(self):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        vocab_size = 556 if self.ds_size == "small" else 8192
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        print("training the library tokenizer ...")
        start = time.time()
        tokenizer.train_from_iterator(self.train_stories, trainer=trainer)
        print(f"training took: {time.time() - start:.1f} seconds")
        tokenizer.save("tokenizer.json")
        ids = tokenizer.encode(self.heldout_text).ids
        n_bytes = len(self.heldout_text.encode("utf-8"))
        print(f"held-out: {len(ids)} tokens, {n_bytes / len(ids):.3f} bytes/token")

print("#"*45)
print(
    "The tokenizer support two dataset sizes: \n"
    "1. 'small' will run the tokenizer on a tiny sample from TinyStories dataset \n"
    "2. 'full' run the tokenizer on a complete training set from TinyStories dataset"
)
print(
    "The tokenizer support two tokenizer types: \n"
    "1. 'own' will run our own tokenizer on the selected dataset\n"
    "2. 'lib' run the library tokenizer on the selected dataset"
)
print("#"*45)
ds_size = input("please select dataset size (small/full): ")
tok_type = input("please select the tokenizer type (own/lib): ")

my_tok = MyTokenizer(ds_size=ds_size, tok_type=tok_type)
my_tok.load_ds()
my_tok.train()