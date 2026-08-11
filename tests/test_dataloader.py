import torch
import unittest
from torch.utils.data import DataLoader
from tokenizer.character import CharacterTokenizer
from datasets.gpt_dataset import GPTDataset

class TestDataloader(unittest.TestCase):
    def test_dataloader(self):
        text = "Test data"
        tokenizer = CharacterTokenizer(text)
        tokens = tokenizer.encode(text)
        dataset = GPTDataset(tokens, context_length=2)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        for x, y in loader:
            self.assertEqual(x.shape, (2, 2))
            self.assertEqual(y.shape, (2, 2))
            break

            expected_x = torch.tensor([
                tokens[0:2],
                tokens[1:3]
            ])

            expected_y = torch.tensor([
                tokens[1:3],
                tokens[2:4]
            ])

            self.assertTrue(torch.equal(x, expected_x))
            self.assertTrue(torch.equal(y, expected_y))
