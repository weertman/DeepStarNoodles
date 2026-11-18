# DeepStarNoodles methods paper

**DeepStarNoodles: A deep-learning pipeline for high-throughput behavioral analysis of sunflower stars**, prepared for PLOS Computational Biology and bioRxiv.

This is the draft of 17 November 2026. Passages marked **[GAP: …]** are still to be written: the acknowledgments and some details of the data sources and annotation in S1 Text. They will be filled before submission.

| File | Contents |
| --- | --- |
| [`DeepStarNoodles_draft_20261117.pdf`](DeepStarNoodles_draft_20261117.pdf) | Main text with the figures inline |
| [`DeepStarNoodles_S1_Text_draft_20261117.pdf`](DeepStarNoodles_S1_Text_draft_20261117.pdf) | S1 Text: datasets, models, settings, full results, throughput, published model files (Table L) and limits of the evidence |
| [`tex/`](tex/) | LaTeX source (PLOS template v3.8); `main.tex` and `s1_text.tex` include `sections/*.tex` |
| [`figures/`](figures/) | Fig 1 to Fig 8, S1 Fig and S2 Fig as PDF |
| [`figure_scripts/`](figure_scripts/) | The matplotlib scripts that drew the figures |
| [`build.sh`](build.sh) | Builds both PDFs into `build/` with pdflatex and bibtex |

## Build

```bash
cd paper
./build.sh            # build/main.pdf and build/s1_text.pdf
./build.sh --no-figs  # main text without inline figures, as PLOS wants it
```

## Figures

| Figure | Script |
| --- | --- |
| Fig 1 | `build_figure_01.py` |
| Fig 2 | `build_figure_02.py` |
| Fig 3 | `build_figure_03.py` |
| Fig 4 | `build_figure_gru_vs_dijkstra.py` |
| Fig 5 | `build_figure_inference.py` |
| Fig 6 | `build_figure_04.py` |
| Fig 7 | `build_figure_05.py` |
| Fig 8 | `build_figure_06.py` |
| S1 Fig | `build_figure_s1_v3.py` |
| S2 Fig | `build_figure_f1_measurement.py` |

The scripts document how each panel was drawn. Each reads a data folder of model outputs, scores and frames of recordings, prepared from the published models and the frozen datasets; those folders are not included. `published_models.py` names the three model files the figures used.

## Models and data

The models are in the GitHub release [`models-20261117`](https://github.com/weertman/deepstarnoodles-inference/releases/tag/models-20261117) and on Hugging Face ([`weertman/deepstarnoodles`](https://huggingface.co/weertman/deepstarnoodles)); S1 Table L lists their SHA-256 hashes. The datasets are [`weertman/deepstarnoodles-data`](https://huggingface.co/datasets/weertman/deepstarnoodles-data) (arms) and [`weertman/starseg`](https://huggingface.co/datasets/weertman/starseg) (detection). The repository [README](../README.md) gives the commands that trained each model.
