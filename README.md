#Name: Sravya Mendem
#Student ID: 700765926

Transformer Components in PyTorch and NumPy:
This repository contains simple and educational implementations of core Transformer architecture components — specifically Scaled Dot-Product Attention and a Transformer Encoder Block — written from scratch using PyTorch and NumPy.
These scripts are designed for students and researchers who want to understand the mathematical foundations and code structure of Transformer models, which are the backbone of architectures like BERT, GPT, and Vision Transformers.

Files Overview:
1) Compute Scaled Dot-Product Attention (Python)
File: Compute Scaled Dot-Product Attention (Python).py

Description:
A NumPy-based implementation of the Scaled Dot-Product Attention mechanism, the core operation behind Transformer self-attention.

Key Steps Implemented:
Apply softmax to obtain normalized attention weights.
Compute the context vector as a weighted sum of value vectors.

Implement Simple Transformer Encoder Block (PyTorch):
File: Implement Simple Transformer Encoder Block (PyTorch).py

Description:
A minimal yet complete PyTorch implementation of a Transformer Encoder Block, including:
* Multi-Head Self-Attention
* Position-wise Feed-Forward Network (FFN)
* Residual Connections + Layer Normalization

Modules Included:
* MultiHeadSelfAttention
* FeedForward
* SimpleTransformerEncoder

Concepts Illustrated:
* Scaled Dot-Product Attention: foundation of all Transformer models.
* Multi-Head Attention: parallel attention heads for diverse feature subspaces.
* Feed Forward Network (FFN): applies non-linearity and projection.
* Residual + LayerNorm: stabilizes gradients and accelerates training.
