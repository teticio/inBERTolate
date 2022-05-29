---
title: InBERTolate
emoji: 🚀
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 3.0.9
app_file: app.py
pinned: false
license: gpl-3.0
---

# inBERTolate
## Hit your word count by using BERT to pad out your essays!

Sentences are generated that are in context with both the preceding and following sentences. Models like GPT are not well suited to this task as they are Causal Language Models, or autoregressive models, that generate tokens from left to right, conditional on the text that has come before. The B in BERT, on the other hand, stands for "Bidirectional" and it was trained to be able to fill in the gaps using context on either side. BERT is an example of an autoencoder model.

Both BERT and GPT are based on [transformers](https://jalammar.github.io/illustrated-transformer/) - which were originally conceived for Neural Translation and consisted of an encoder and a decoder - but while GPT is a decoder without an encoder, BERT is an encoder without a decoder (the E in BERT). As a result, GPT is a more natural choice for language generation. BERT can be coaxed into generating language by leveraging its ability to fill in the gaps (masked tokens). Done naively this gives disappointing results, but the paper ["BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model"](https://arxiv.org/abs/1902.04094) shows how this can be acheived much more effectively, although much more slowly, as it requires doing a MCMC (Markov Chain Monte Carlo) simulation. I have made some minor adjustments to take into account left and right context as well as to use the HuggingFace package. I also modified it to use RoBERTa large.

I have deployed it as a simple web app on [HuggingFace spaces](https://huggingface.co/spaces/teticio/inBERTolate). Without a GPU, however, it is very slow. If it is a bit too random, try reducing the temperature.
