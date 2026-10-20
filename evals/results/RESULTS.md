## Results

| System | Top-1 | CER | Top-5 | ms / word |
|---|---|---|---|---|
| rule engine (ours) | 9.2% | 48.8% | 13.6% | 0.10 |
| our model | 53.3% | 17.1% | 72.2% | 24.49 |

### Breakdown by test subset

| Test set | AK-Freq | AK-NEF | AK-NEI | Dakshina |
|---|---|---|---|---|
| Top-1 | 58.0% | 25.3% | 28.5% | 58.7% |
| CER | 11.9% | 50.0% | 41.1% | 15.4% |
| words | 4858 | 833 | 1191 | 5569 |

### Does the augmentation help?

| Training data | Standard test set | | Adhoc test set | |
|---|---|---|---|---|
| standard rows only | 56.3% | 15.4% | 6.1% | 50.0% |
| standard + adhoc rows | 53.3% | 17.1% | 46.2% | 20.1% |

