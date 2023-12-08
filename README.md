# Token-Transfer-Learning
Music, like words, has inherent semantics and meaning. The composition, order of the notes, are tied to the expressiveness of the music piece such as genre, tempo, instrumentation and more. Understanding and replicating semantics are crucial for generating music that is coherent and emotionally resonant. We present a music generation model leveraging different neural network architectures to extract informative embedding and employ Markov models to comprehend and replicate the semantics of music, facilitating the creation of coherent and emotionally resonant musical compositions.


Model steps are 
1. Tokenize new midi files (MIDI files available at https://github.com/asigalov61/Tegridy-MIDI-Dataset)
2. Use gpu_extract_embeddings_from_forward.ipynb to get positional and token embeddings
3. Use make_new_token_adj.ipynb to make adjacency matrix using token neighbors in new tokenized data
4. Run GCN.py to train GCN using transformer embeddings and RWR to generate new sequences
