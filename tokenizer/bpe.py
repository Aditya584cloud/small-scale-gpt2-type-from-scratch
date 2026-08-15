import json


class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i <len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):

        ids = list(text.encode("utf-8"))
        x = self.vocab_size - 256
        for i in range(x):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key = stats.get)
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = self.merge(ids, pair, new_id)
            
            self.merges[pair] = new_id

    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        while(len(tokens)>1):
            best_pair = None
            best_id = float('inf')

            for pair in zip(tokens, tokens[1:]):
                if pair in self.merges:
                    new_id = self.merges[pair]

                    if new_id < best_id:
                        best_id = new_id
                        best_pair = pair
            if best_pair is None:
                break
            tokens = self.merge(tokens, best_pair, best_id)
        return tokens

    def decode(self, ids):
        text = b""
        for idx in ids:
            text += self.vocab[idx]
        
        return text.decode("utf-8")

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [
                [list(pair), new_id]
                for pair, new_id in self.merges.items()
            ],
            "vocab": {
                str(idx): value.hex()
                for idx, value in self.vocab.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tokenizer = cls(data["vocab_size"])

        for pair, new_id in data["merges"]:
            tokenizer.merges[tuple(pair)] = new_id

        for idx, value in data["vocab"].items():
            tokenizer.vocab[int(idx)] = bytes.fromhex(value)

        return tokenizer
        
