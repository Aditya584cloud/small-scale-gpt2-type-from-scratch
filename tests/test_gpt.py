import torch
import unittest

from model.gpt import GPT,AttentionHead, Block, GPTConfig

class TestGPT(unittest.TestCase):

    def setUp(self):
        self.config = GPTConfig(vocab_size=50, context_length=16, embedding_dim=128, num_heads=4, num_layers=4)
        self.model = GPT(self.config)

    def test_embedding_output_shape(self):
        x = torch.randint(0, self.config.vocab_size, (2, self.config.context_length))
        output = self.model(x)
        self.assertEqual(output.shape, (2, self.config.context_length, self.config.vocab_size))

    def test_masking(self):
        config = GPTConfig(vocab_size=50, context_length=16, embedding_dim=128, num_heads=1, num_layers=4)

        attention = AttentionHead(config)

        x = torch.randn(1, config.context_length, config.embedding_dim)
        _, wei = attention(x)

        future = torch.triu(
            wei,
            diagonal=1
        )

        self.assertTrue(torch.all(future == 0))

    def test_attention_head_output_shape(self):
        config = GPTConfig(vocab_size=50, context_length=16, embedding_dim=128, num_heads=4, num_layers=4)
        attention = Block(config)

        x = torch.randn(2, config.context_length, config.embedding_dim)
        output = attention(x)
        self.assertEqual(output.shape, (2, config.context_length, config.embedding_dim))

    def test_gpt_output_shape(self):
        x = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.context_length)
        )

        logits = self.model(x)

        self.assertEqual(
            logits.shape,
            (2, self.config.context_length, self.config.vocab_size)
        )
    
if __name__ == '__main__':
    unittest.main()

