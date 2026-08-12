import torch
import unittest

from model.gpt import GPT,AttentionHead, Block, GPTConfig

class TestGPT(unittest.TestCase):

    def setUp(self):
        self.config = GPTConfig(vocab_size=50, context_length=16, embedding_dim=128, num_heads=4, num_layers=4)
        self.model = GPT(self.config)

    def test_embedding_output_shape(self):
        x = torch.randint(0, self.config.vocab_size, (2, self.config.context_length))
        output, _ = self.model(x)
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

        logits, loss = self.model(x, targets=x)

        self.assertEqual(
            logits.shape,
            (2, self.config.context_length, self.config.vocab_size)
        )
    
    def test_loss(self):
        x = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.context_length)
        )

        y = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.context_length)
        )

        logits, loss = self.model(x, targets=y)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_gpt_with_targets(self):
        x = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.context_length)
        )

        targets = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.context_length)
        )

        logits, loss = self.model(x, targets)

        self.assertEqual(
            logits.shape,
            (2, self.config.context_length, self.config.vocab_size)
        )

        self.assertTrue(torch.isfinite(loss))

    def test_training_step(self):
        x = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.context_length)
        )

        targets = torch.randint(
            0,
            self.config.vocab_size,
            (2, self.config.context_length)
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3
        )

        before = self.model.token_embedding.weight.detach().clone()

        logits, loss = self.model(x, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        after = self.model.token_embedding.weight.detach()

        self.assertFalse(torch.equal(before, after))
        
        
if __name__ == '__main__':
    unittest.main()

