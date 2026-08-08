import unittest
from tokenizer.character import CharacterTokenizer

class TestCharacterTokenizer(unittest.TestCase):
    def test_vocab_size(self):
        text = "hello"
        tokenizer = CharacterTokenizer(text)
        self.assertEqual(tokenizer.vocab_size, 4)

    def test_encode_decoded(self):
        text = "hello"
        tokenizer = CharacterTokenizer(text)
        
        self.assertEqual(
            tokenizer.encode("hello"), 
            [1, 0, 2, 2, 3]
            )

        encoded = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(encoded), text)
    
    def test_unknown_character(self):
        text = "hello"
        tokenizer = CharacterTokenizer(text)
        
        with self.assertRaises(ValueError):
            tokenizer.encode("world")

    def test_empty_string(self):
        text = "hello"
        tokenizer = CharacterTokenizer(text)
        
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.decode([]), "")

    def round_trip_test(self, text):
        text = "hello World"
        tokenizer = CharacterTokenizer(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)
