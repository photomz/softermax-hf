# Overview

A test of the ['Attention is Off by One'](https://www.evanmiller.org/attention-is-off-by-one.html) hypothesis.
Softermax implementation of llama based on HuggingFace Transformers.
We name the subclassed huggingface model SofterLlama (mainly so IDE autocomplete is more convenient than "LlamaSoftermax", but it also sounds nicer).

# Devlog
We track development using this [Google Doc](https://docs.google.com/document/d/1cEiQyOfQDFaNUyDZDg5TjE-439I4S_a2CyiyA9Fuc9Y/edit?usp=sharing)


# Requirements

Coincidentally, our timing on adapting llama falls under a [massive PR](https://github.com/huggingface/transformers/pull/26681) refactoring kv cache utils, partially motivated by
[attention sinks](http://arxiv.org/abs/2309.17453) which is tangentially related to softermax (main difference being our inital "token" that serves as an 
"attention sink" doesn't have a corresponding V matrix entry).

Since this is the case, we must use transformers==4.36.1 onwards as the latest version to have access to the Cache object.
