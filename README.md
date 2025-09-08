# Charting the Future: Harnessing LLMs for Quantitative Finance
[![](https://github.com/charting-the-future/charting-the-future/actions/workflows/ci.yml/badge.svg)](https://github.com/charting-the-future/charting-the-future/actions/workflows/ci.yml) 
![](https://img.shields.io/badge/python-3.13-blue.svg)

Welcome to the companion repository for the book  
**Charting the Future: Harnessing LLMs for Quantitative Finance** (Colin Alexander, CFA, CIPM).

This repository provides:
- **Book Manuscript** (chapters, figures, references, teaching material)  
- **Working Code Examples** (sector-committee, portfolio management, labs)  
- **Research Summaries** (structured notes on key LLM + finance papers)  
- **Instructor Pack** (slides, labs, solutions for classroom adoption)  

The goal is to bridge **theory → practice**: every concept in the book is tied to reproducible code, data pipelines, and academic research.

---

## Repository Structure

```text
charting-the-future/
├─ book/                         # Book manuscript, figures, references
│  ├─ src/
│  │  ├─ chapters/               # Drafts and edited chapters
│  │  ├─ figures/                # Diagrams and illustrations
│  │  └─ references/             # Consolidated and per-chapter references
│  ├─ marketing/                 # Book marketing collateral
│  └─ styles/                    # Templates (Wiley-compliant)
│
├─ sector-committee/             # Capstone code: agents → scores → beta-neutral portfolio
│  ├─ sector_committee/          # Installable Python package
│  ├─ notebooks/                 # End-to-end walkthroughs
│  ├─ configs/                   # Sector mappings, risk limits
│  ├─ tests/                     # Unit tests for agents, tilts, hedging
│  └─ examples/                  # Minimal scripts (weekly run, export orders)
│
├─ research/                     # Research papers referenced in the book
│  ├─ sources.md                 # Master index: citation + URL
│  ├─ summaries/                 # One markdown summary per paper
│  │  ├─ template_summary.md      # Contributor template
│  │  ├─ 2023_tradinggpt_summary.md
│  │  ├─ 2023_alpha_gpt_summary.md
│  │  ├─ 2024_hybridrag_summary.md
│  │  ├─ 2024_quantagent_summary.md
│  │  └─ ...
│  └─ figures/                   # Redrawn charts or diagrams from papers
│
├─ datasets/                     # Small public sample data for notebooks
│  ├─ prices/                    # Sample OHLCV data for SPDR ETFs
│  ├─ macro/                     # Example macroeconomic indicators
│  └─ transcripts/               # Short excerpts for text processing demos
│
├─ instructor-pack/               # Teaching kit for classroom adoption
│  ├─ labs/                       # Hands-on labs (e.g., sector committee lab)
│  ├─ slides/                     # Chapter-by-chapter slide decks
│  └─ solutions/                  # Release-gated and will not be accessible in public versions of the repo
│
├─ docs/                         # Central documentation
│  ├─ architecture.md             # System diagrams
│  ├─ governance.md               # Model governance, audit mapping (SR 11-7, MiFID II)
│  └─ api/                        # API reference for sector committee
│
├─ website/                      # (Optional) Static site for the project
│
├─ .github/                      # GitHub configs, CI/CD, PR templates
├─ .gitignore                    # Gitignore file (Python template)
├─ CODE_OF_CONDUCT.md            # Contributor guidelines
├─ LICENSE                       # MIT license
├─ README.md                     # (this file)
├─ SECURITY.md                   # Security policies
└─ pyproject.toml                # Project file for uv
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/charting-the-future/charting-the-future.git
cd charting-the-future
```

### 2. Set up environment with [`uv`](https://docs.astral.sh/uv/)

We use `uv` for fast Python environment and dependency management. `uv` is a Rust-based, ultra-fast Python package and project manager. Refer to uv's [installation](https://docs.astral.sh/uv/getting-started/installation/) guide.
Note: Mac users with [homebrew](https://brew.sh/) can simply run `brew install uv`.

The project targets Python 3.13.

From the project root directory, run:
```bash
# Sync dependencies (from pyproject.toml)
uv sync
```

This will create a `.venv` folder and install all dependencies in an isolated environment. 
Note: you don’t need to activate the environment manually — always run commands with `uv run ...`, e.g. `uv run my_module.py`.  It is an alias for `uv run python ...`.

### 3. Install the `sector-committee/` package
```bash
cd sector-committee
uv pip install -e .
```

### 4. Lint and test the code
```bash
uv run ruff check .    # run linter on current directory
uv run ruff format .   # format code
uv run pytest  # run test suite
```

### 5. Run the capstone notebook
```bash
uv run jupyter notebook notebooks/00_quickstart.ipynb
```

This walks through:  
agents → sector scores (1–5) → committee aggregation → long/short portfolio → beta-hedge.

---

## Research Summaries

The `research/` folder contains structured 1–2 page summaries of key LLM + finance papers.  
- Use `template_summary.md` for new submissions.  
- Each file is named `<year>_<shorttitle>_summary.md`.  
- All sources are indexed in `sources.md` with DOI/arXiv/SSRN links.

---

## Contributing

This repository is provided as a companion to the book and is not open for general contributions.  
Readers and students are encouraged to fork the repository for their own experimentation and extensions.  
If you notice an error (e.g., broken link, typo, or reproducibility issue), please open an [Issue](https://github.com/charting-the-future/charting-the-future/issues) instead of submitting a pull request.  Refer to `SECURITY.md` for submitting security vulnerabilities.
Please note that the official repository will only be updated by the author.

---

## Disclaimer

This repository and its contents are provided for **educational and research purposes only**.  
Nothing in this repository — including code, notebooks, or written material — should be construed as financial, investment, legal, or other professional advice.  

Use of the examples is entirely at your own risk.  
The author and publisher assume no responsibility for any financial losses, damages, or other consequences resulting from the use of the materials provided here.

---

## License

MIT License for code.  
Book content © 2025 Colin Alexander.  
Figures and text may be reused only with attribution.  
See [SECURITY.md](SECURITY.md) for the security policy.

---

## Citation

If you use this repo in your research or teaching, please cite:

> Alexander, C. (2025). *Charting the Future: Harnessing LLMs for Quantitative Finance*. Wiley (forthcoming).  
