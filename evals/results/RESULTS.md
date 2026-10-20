## Results

| System | Top-1 | CER | Top-5 | ms / word |
|---|---|---|---|---|
| rule engine (ours) | 9.2% | 48.8% | 13.6% | 0.10 |
| our model | 53.4% | 17.0% | 64.6% | 27.40 |
| our model, no augmentation | 56.2% | 15.4% | 67.3% | 4.01 |

### Breakdown by test subset

| Test set | AK-Freq | AK-NEF | AK-NEI | Dakshina |
|---|---|---|---|---|
| Top-1 | 58.2% | 25.2% | 29.5% | 58.5% |
| CER | 11.8% | 49.7% | 40.3% | 15.4% |
| words | 4858 | 833 | 1191 | 5569 |

### Does the augmentation help?

| Training data | Standard test set | | Adhoc test set | |
|---|---|---|---|---|
| standard rows only | 56.2% | 15.4% | 6.2% | 49.7% |
| standard + adhoc rows | 53.4% | 17.0% | 46.2% | 20.0% |

