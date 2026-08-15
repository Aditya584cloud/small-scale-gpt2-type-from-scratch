import unittest
from datasets.gpt_dataset import GPTDataset
import torch

class TestGPTDataset(unittest.TestCase):
    def setUp(self):
        self.tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.context_length = 3
        self.dataset = GPTDataset(self.tokens, self.context_length)

    def test_len(self):
        expected_length = len(self.tokens) - self.context_length
        self.assertEqual(len(self.dataset), expected_length)

    def test_first_sample(self):
        x, y = self.dataset[0]
        expected_x = torch.tensor([1, 2, 3], dtype=torch.long)
        expected_y = torch.tensor([2, 3, 4], dtype=torch.long)
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_second_sample(self):
        x, y = self.dataset[1]
        expected_x = torch.tensor([2, 3, 4], dtype=torch.long)
        expected_y = torch.tensor([3, 4, 5], dtype=torch.long)
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))
        self.assertTrue((y == torch.tensor([3, 4, 5])).all())

    def test_last_sample(self):
        x, y = self.dataset[len(self.dataset) - 1]
        expected_x = torch.tensor([7, 8, 9], dtype=torch.long)
        expected_y = torch.tensor([8, 9, 10], dtype=torch.long)
        self.assertTrue(torch.equal(x, expected_x))
        self.assertTrue(torch.equal(y, expected_y))

    def test_out_of_bounds(self):
        with self.assertRaises(IndexError):
            _ = self.dataset[len(self.dataset)]