class CharacterTokenizer:
    def __init__(self, text):
        self.text = text
        self.vocab = sorted(set(self.text))
        self.char_to_idx = {
            char: idx 
            for idx, char in enumerate(self.vocab)
            }
        self.idx_to_char = {
            idx: char 
            for idx, char in enumerate(self.vocab)
            }

    def encode(self, text):
        try:
            return [self.char_to_idx[char] for char in text]
        except KeyError as e:
            raise ValueError("Unknown character: {char}".format(char=e.args[0]))
    
    def decode(self, ids):
        try:
            return ''.join(self.idx_to_char[idx] for idx in ids)
        except KeyError as e:
            raise ValueError("Unknown index: {idx}".format(idx=e.args[0]))

    def vocab_size(self):

   