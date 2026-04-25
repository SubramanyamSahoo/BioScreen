# 🧬🛡️ BioScreen

**Function-aware DNA synthesis screening against AI-designed biological threats**

[![AIxBio 2026](https://img.shields.io/badge/AIxBio-Hackathon_2026-blue)](https://apartresearch.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)

## The Problem

Current DNA synthesis screening relies on **sequence homology** — matching against
databases of known threats. AI protein design tools (ProteinMPNN, RFdiffusion, Evo2)
can generate **functional variants with <40% sequence identity** to known pathogens,
evading all homology-based screening (Microsoft et al., *Science*, Oct 2025).

## Our Solution

BioScreen uses **supervised contrastive fine-tuning of ESM-2 (3B)** to learn a
functional embedding space where proteins cluster by **mechanism of harm**, not
sequence similarity.

| Method | Detection @ <40% ID | AUROC |
|--------|---------------------|-------|
| k-mer (BLAST proxy) | 65.8% | 0.785 |
| Raw ESM-2 similarity | 86.9% | 0.967 |
| **BioScreen Classifier** | **99.3%** | **0.998** |
| **BioScreen Embedding** | **99.0%** | **0.997** |

## Key Results

- **4,981** protein sequences, **7 threat mechanisms** + benign
- **2,500** adversarial variants via ESM-2 MLM-guided mutation
- **90.8%** mechanism-of-harm classification accuracy (8 classes)
- **44.4 seq/s** throughput on NVIDIA H200 (370x headroom)
- Production REST API with Docker deployment

## Quick Start

### Python API

```python
from bioscreen import BioScreener

screener = BioScreener.from_pretrained('bioscreen-v1')
results = screener.screen(['MKFLVLLF...'])

for r in results:
    print(f'{r.decision}: {r.predicted_mechanism}')
    print(f'  Risk: {r.risk_score:.2f} +/- {r.risk_uncertainty:.2f}')
```

### REST API

```bash
# Start server
docker compose -f docker/docker-compose.yml up

# Screen a sequence
curl -X POST http://localhost:8080/v1/screen \\
  -H 'Content-Type: application/json' \\
  -d '{"sequences": ["MKFLVLLF..."]}'
```

### Install from source

```bash
git clone https://github.com/YOUR_USERNAME/bioscreen.git
cd bioscreen
pip install -e .
```

## Architecture

```
Protein Sequence
  → ESM-2 3B (30/36 layers frozen, fp16)
  → Functional Embedding (256-dim, L2-normalized)
    ├→ Binary: PASS / REVIEW / BLOCK
    ├→ Mechanism: 7 threats + benign
    ├→ Risk Score: [0,1] + uncertainty
    └→ Contrastive projection (training only)
```

## Ablation Study

| Variant | AUROC | Det @ <40% ID |
|---------|-------|---------------|
| Contrastive + PGD Adversarial | 0.958 | 72.6% |
| **Contrastive Only** | **0.980** | **96.2%** |
| No Contrastive | 0.848 | 48.7% |
| Frozen ESM-2 | 0.781 | 25.7% |

**Key finding:** PGD adversarial training in embedding space *hurts* detection.
Contrastive learning alone is the critical component.

## Citation

```bibtex
@misc{bioscreen2026,
  title={BioScreen: Function-Aware DNA Synthesis Screening},
  author={AIxBio Hackathon Team},
  year={2026},
  note={AIxBio Hackathon, Track 1}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
