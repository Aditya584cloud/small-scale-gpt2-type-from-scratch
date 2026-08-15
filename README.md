# Small-Scale GPT-2-Type Language Model from Scratch

A small decoder-only Transformer language model built from scratch in PyTorch.

The goal of this project is to understand and implement the core components behind GPT-style language models rather than relying on high-level Transformer libraries.

## Project Goals

- Implement a byte-level BPE tokenizer from scratch
- Build a decoder-only Transformer architecture
- Train the model on TinyStories
- Implement autoregressive text generation
- Understand the complete training and inference pipeline
- Benchmark and optimize the implementation
- Gradually scale the model and study how performance changes

## Current Architecture

| Component | Configuration |
|---|---:|
| Vocabulary | 1024 |
| Context length | 128 |
| Embedding dimension | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Parameters | ~1M |

## Tokenizer

The project currently uses a byte-level BPE tokenizer implemented from scratch.

Features:

- 256 base byte tokens
- Learned BPE merges
- Vocabulary size: 1024
- Encode / decode
- Tokenizer serialization
- Cached tokenized datasets

## Dataset

The current model is trained on a sample of the TinyStories dataset.

The tokenized corpus is stored as `uint16` binary data and accessed using memory mapping for efficient random sampling.

## Training

The model is trained autoregressively using next-token prediction:

```text
Input:  t0  t1  t2  t3 ... t(n-1)
Target: t1  t2  t3  t4 ... tn
```

Optimizer:

- AdamW

Current training configuration:

- Batch size: 32
- Context length: 128
- 500 sampled batches per epoch
- 20 epochs

## Initial Results

The first end-to-end BPE-GPT training run achieved:

| Metric | Result |
|---|---:|
| Training loss | 2.9528 |
| Validation loss | 2.8273 |

The trained model successfully generates TinyStories-style text from prompts.

Example:

```text
Prompt:
Once upon a time

Generated:
Once upon a time, a little bird named Tim...
```

## Project Status

### Completed

- [x] Character tokenizer
- [x] Byte-level BPE tokenizer
- [x] BPE training
- [x] BPE encode/decode
- [x] Tokenizer save/load
- [x] TinyStories preprocessing
- [x] Binary token storage
- [x] Memory-mapped random batch sampling
- [x] GPT training
- [x] Validation
- [x] Checkpoint saving
- [x] Autoregressive generation

### In Progress

- [ ] Temperature sampling
- [ ] Top-k sampling
- [ ] Top-p sampling
- [ ] Cleaner generation API
- [ ] Better validation methodology
- [ ] BPE encoding optimization
- [ ] BPE training optimization
- [ ] Training pipeline profiling
- [ ] Model scaling experiments
- [ ] Systematic evaluation

## Architecture

```text
                    TinyStories
                         |
                         v
                Byte-Level BPE
                         |
                         v
                   Token IDs
                         |
                         v
                  GPT Data Sampler
                         |
                         v
                  Decoder-Only GPT
                         |
              +----------+----------+
              |                     |
        Token Embedding       Position Embedding
              |                     |
              +----------+----------+
                         |
                         v
                 Transformer Blocks
                         |
                         v
                    LayerNorm
                         |
                         v
                    LM Head
                         |
                         v
                 Next-Token Logits
                         |
                         v
                    Sampling
                         |
                         v
                    BPE Decode
                         |
                         v
                     Text
```

## Why "GPT-2-Type"?

This project is inspired by the architecture and training paradigm of GPT-style decoder-only Transformers, particularly GPT-2.

It is **not a reproduction of the original GPT-2 model or its scale**.

The initial implementation is intentionally small so that the entire system can be understood, implemented, profiled, and experimented with from scratch.

## Future Engineering Direction

The project will progressively explore:

1. Efficient tokenizer implementations
2. Training-system optimization
3. Larger model configurations
4. Generation quality
5. Evaluation and ablations
6. Training stability
7. Scaling behavior
8. Modern improvements to the GPT architecture

The emphasis is on understanding the engineering and mathematical principles behind the system rather than simply obtaining a working model.

## Research Direction

Beyond building and scaling the model, this project will serve as a foundation for studying the internal behavior of Transformer language models.

Planned research directions include:

1. **Representation analysis**
   - Track how representations evolve across Transformer layers
   - Study where semantic information emerges and transforms
   - Compare representations across different inputs expressing similar meanings

2. **Mechanistic interpretability**
   - Analyze attention heads, MLPs, residual streams, and intermediate activations
   - Investigate which components are causally involved in specific model behaviors
   - Study distributed representations rather than assuming concepts correspond to individual neurons

3. **Information flow**
   - Analyze how information propagates through the network
   - Study activation-space and Jacobian-based representations
   - Investigate when information becomes causally available for downstream predictions

4. **Causal experiments**
   - Activation patching
   - Representation interventions
   - Layer-wise ablations
   - Causal tracing of model behavior

5. **Scaling experiments**
   - Compare representation and information-flow behavior across model sizes
   - Study how qualitative behaviors emerge as model capacity increases
   - Relate internal changes to changes in training loss and model capabilities

The long-term goal is to use increasingly capable versions of this from-scratch GPT as controlled experimental systems for understanding how Transformer models represent, transform, and use information.
