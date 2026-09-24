# H5: geometry of vocabulary blocks

`run.py` (exp440) runs the character corpus through GPT-2 BPE, remapped to the ~11.7k token types that actually occur in it, so rare tokens are real. It compares Adam against per-token rows on embed/pos/head and reports the loss per train-frequency decile of the target token. This directory was added because H5 needs its own data pipeline.
