# Transformer from Scratch
Implement the Transformer model from scratch and train it on English - Spanish translation task

This repository contains an implementation of a machine translation model based on the revolutionary ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf) paper, commonly known as the Transformer architecture. The model is trained to translate English sentences into Spanish sentences using a neural network approach that focuses on self-attention mechanisms.

# Usage
```
pip install -r requirements.txt

$ python translate.py "Hello, my name is John."
>>> Hola , mi nombre es John .
```

# Features
- Transformer architecture implemented from scratch in PyTorch.
- The model trained from scratch on [English - Spanish dataset](https://www.manythings.org/anki/spa-eng.zip).

# Output
![image](https://github.com/tonystark11/transformer-from-scratch/assets/20776426/fb394d44-5905-4168-840f-fb82e4c2d622)

# References
1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
2. [Build your own Transformer from scratch using PyTorch](https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb)
3. [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
