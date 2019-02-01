#!/usr/bin/env python
# encoding: utf-8

import numpy as np
wordvec = {}

with open("/dataset/embeddings/GoogleNews-vectors-negative300.bin", "rb") as f:
    header = f.readline()
    ht = header.strip().split()
    vocab_size, embedding_size = int(ht[0]), int(ht[1])
    # print embedding_size

    binary_len = np.dtype(np.float32).itemsize * embedding_size

    for _ in xrange(vocab_size):
        word = []
        while True:
            ch = f.read(1)
            if ch == " ":
                break
            if ch != "\n":
                word.append(ch)
        vec = np.fromstring(f.read(binary_len), dtype=np.float32).astype(np.float32)
        w = "".join(word)
        wordvec[w] = vec


