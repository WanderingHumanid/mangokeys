# Credits & Licensing

Everything this project uses, its licence, how we use it, and what we owe.

> **Academic use.** Every asset below permits academic use. Two carry share-alike obligations
> (Dakshina, and the Ezhuthola fork if you use it) — that just means your redistribution uses
> the same licence. **No legal review is needed.**

---

## Summary

| Asset | Licence | Role | Obligation |
|---|---|---|---|
| [Dakshina](https://github.com/google-research-datasets/dakshina) (Google) | **CC BY-SA 4.0** | Training + evaluation | Attribute; share-alike on redistribution |
| [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) (AI4Bharat) | CC-BY (manual) / CC0 (mined) | Training + evaluation | Attribute the manual rows |
| [IndicXlit](https://github.com/AI4Bharat/IndicXlit) model | MIT | Baseline | Include MIT notice |
| [IndoNLP-2025](https://aclanthology.org/2025.indonlp-1.20) test sets | Per task | Evaluation only | Cite; **never train on** |
| [Mozhi 2.0](https://mozhi-documentation.digitalmalayali.in) spec | Open spec | Rule engine + conformance tests | Cite |
| [Ezhuthola](https://github.com/rag-creation/ezhuthola) (Numio) — if forked | **GPL-3.0** | Android shell + **rule engine** | Fork must be GPL-3.0. **Its `ezhuthola_rules.json` and `MalayalamEngine.kt` are GPL-3.0 too** — reusable in a GPL project, not in a permissively-licensed one |
| [Indic Keyboard](https://gitlab.com/indicproject/indic-keyboard) (SMC) — if forked | Apache-2.0 | Android shell | Keep NOTICE, state changes |
| [Aksharamukha](https://github.com/virtualvinodh/aksharamukha-python) | GPL-3.0 | Reference only | Do not vendor into non-GPL code |
| CMU Pronouncing Dictionary | BSD-3-Clause | (Stretch) loanword mapping | Include notice |
| Noto Sans Malayalam | SIL OFL 1.1 | UI font | Ship unmodified + include OFL |
| ONNX Runtime | MIT | Inference runtime | Include notice |
| PyTorch | BSD-3-Clause | Training | Not shipped |
| **Our model weights** | **CC BY-SA 4.0** (recommended) | The artifact | Attribute; share-alike |

---

## 1. Dakshina (Google) — primary dataset

- **Repo:** `github.com/google-research-datasets/dakshina` (README only — data is in
  `dakshina_dataset_v1.0.tar`)
- **Licence:** **CC BY-SA 4.0**
- **Paper:** Roark et al., *Processing South Asian Languages Written in the Latin Script: the
  Dakshina Dataset*, LREC 2020. `aclanthology.org/2020.lrec-1.294/`
- **Use:** training data (`ml/lexicons/`, `ml/romanized/`) and evaluation
  (`ml/lexicons/*.dev.tsv`, `*.test.tsv`)
- **Obligation:** attribute the dataset and the paper. If you redistribute the derived data or
  the data-processing code, do so under CC BY-SA 4.0. **CC BY-SA 4.0 is compatible with GPL-3.0**,
  so it composes cleanly with an Ezhuthola fork.

**Cite:**
```bibtex
@inproceedings{roark-etal-2020-processing,
    title     = {Processing {S}outh {A}sian Languages Written in the Latin Script:
                 the Dakshina Dataset},
    author    = {Roark, Brian and Wolf-Sonkin, Lawrence and Kirov, Christo and
                 Mielke, Sabrina J. and Johny, Cibu and Demir{\c{s}}ahin, I{\c{s}}in and
                 Hall, Keith},
    booktitle = {Proceedings of the 12th Language Resources and Evaluation Conference},
    year      = {2020},
    pages     = {2413--2423},
    url       = {https://aclanthology.org/2020.lrec-1.294/}
}
```

## 2. Aksharantar (AI4Bharat) — volume

- **Repo:** `huggingface.co/datasets/ai4bharat/Aksharantar` (Malayalam slice `ml.zip`, ~125 MB)
- **Licence:** *manually collected* → **CC-BY**; *mined* (Samanantar, IndicCorp) → **CC0**;
  *existing sources* → **CC0**
- **Paper:** Madhani et al., *Aksharantar: Open Indic-language Transliteration datasets and
  models for the Next Billion Users*, Findings of EMNLP 2023.
  `aclanthology.org/2023.findings-emnlp.4/`
- **Use:** training + evaluation
- **Obligation:** attribute AI4Bharat and the paper. Keep the per-row `src` field in processed
  data so CC-BY vs. CC0 attribution stays scoped precisely.

```bibtex
@article{Madhani2022AksharantarTB,
    title   = {Aksharantar: Towards building open transliteration tools for the next billion users},
    author  = {Madhani, Yash and Parthan, Sushane and Bedekar, Priyanka A. and
               Khapra, Ruchi and Seshadri, Vivek and Bedekar, Priyanka and
               Kunchukuttan, Anoop and Kumar, Pratyush and Shantadevi Khapra, Mitesh},
    journal = {arXiv},
    year    = {2022},
    eprint  = {2205.03018},
    url     = {https://arxiv.org/abs/2205.03018}
}
```

## 3. IndicXlit — baseline model

- **Repo:** `github.com/AI4Bharat/IndicXlit` · **Model:** `huggingface.co/ai4bharat/IndicXlit`
- **Licence:** MIT (code **and** weights)
- **Use:** the baseline we must beat; ONNX source for the head-to-head comparison
- **Obligation:** include the MIT notice in the repo, the report, and the app's OSS notices
- **Note:** the ~47 MB ONNX bundle in the community Node port
  (`@cloudrumbles/indic-transliterate`) is an unofficial derivative — its own terms apply there.

## 4. IndoNLP-2025 shared task test sets — evaluation only

- **Paper:** Bajiyo et al., *Romanized to Native Malayalam Script Transliteration Using an
  Encoder-Decoder Framework*, IndoNLP 2025. `aclanthology.org/2025.indonlp-1.20/` ·
  arXiv:2412.09957
- **Use:** **the north-star evaluation.** Test Set-1 (standard) and Test Set-2 (adhoc).
  Published Bi-LSTM baseline: 7.4% / 22.7% CER.
- **Obligation:** cite whenever we report numbers on these sets. **Never train on them** — they
  are the benchmark, and training on them invalidates the whole project.

## 5. Mozhi 2.0 — rule engine + conformance tests

- **Spec:** `mozhi-documentation.digitalmalayali.in` (Rajan Nair & Junaid P. V.; v2.0 by Cibu Johny)
- **Use:** the deterministic rule layer for the live preview and the baseline; its published test
  cases become our `test-mozhi` conformance suite
- **Status:** an open specification, not a licence-restricted dataset. **Cite the spec.**
- **Also used by:** Google Input Tools (Chrome), Indic Keyboard's "Transliteration (Mozhi)"
  layout, Keyman

## 6. Existing IME shells (optional, for the Android demo)

| Project | Licence | Consequence |
|---|---|---|
| [Ezhuthola](https://github.com/rag-creation/ezhuthola) (`com.rr.numio.ezhuthola`) | **GPL-3.0** | A fork is a **GPL-3.0 derivative**; release the fork under GPL-3.0. The rule engine (`ezhuthola_rules.json`, `MalayalamEngine.kt`, `SuffixSplitter.kt`) is GPL-3.0 as well. Fine for academic use and compatible with CC BY-SA 4.0. |
| [Indic Keyboard](https://gitlab.com/indicproject/indic-keyboard) | **Apache-2.0** | Fork-friendly. Keep the NOTICE file and state what you changed. |

**Decide before week 6.** If the fork is only for the demo, Ezhuthola is the faster choice (it is
already a Manglish keyboard, and its rule engine is the candidate generator for the
rules-propose/model-rank architecture). If you want a permissive release, use Indic Keyboard —
but then you must reimplement the rule engine yourself rather than copying Ezhuthola's.

**If you use Ezhuthola's rules in a project that is *not* GPL-3.0:** reimplement them from the
Mozhi spec + the Unicode chillu mapping rather than copying the JSON. The file is small and
readable, so this is a day of work, not a month.

## 7. Aksharamukha — reference only

- **Licence:** GPL-3.0. 120 scripts, 21 romanization schemes, incl. "Roman (Colloquial)".
- **Do not vendor its code** into a non-GPL project. Use it as a behavioural reference for rule
  design and to cross-check outputs.

## 8. Fonts & data

| Asset | Licence | Obligation |
|---|---|---|
| Noto Sans Malayalam | SIL OFL 1.1 | Ship unmodified; include OFL |
| Baloo Chettan 2, Manjari, Gayathri, Chilanka | SIL OFL 1.1 | Same |
| Unicode CLDR + Unicode Emoji data | Unicode License v3 | Include notice |
| CMU Pronouncing Dictionary | BSD-3-Clause | Include notice |

## 9. Our trained model

**Recommended licence: CC BY-SA 4.0**, because we train on Dakshina (CC BY-SA 4.0). It keeps
everything consistent and is compatible with the GPL-3.0 Android fork.

If you decide to train on Aksharantar only, MIT becomes possible — but check with your
department/instructor first, since the Dakshina data will still be in your eval scripts.

---

## Checklist before you present

- [ ] `CREDITS.md` or a Credits section in the README + `REPORT.md`, with every dataset, model,
      and paper
- [ ] BibTeX entries for Dakshina, Aksharantar, Bajiyo et al., IndicXlit, and Mozhi
- [ ] `LICENSE` file at the repo root matching what you chose
- [ ] Model card (even a short one) on Hugging Face, if you publish weights
- [ ] If you forked a keyboard: keep its LICENSE and NOTICE, and state your changes
- [ ] Confirm the shipped app requests **no** `INTERNET` permission (nice line for the poster:
      *"no permissions at all — nothing you type leaves your phone"*)
- [ ] Mention the licences in the report's ethics/credits section — it takes one paragraph and
      reads as unusually professional
