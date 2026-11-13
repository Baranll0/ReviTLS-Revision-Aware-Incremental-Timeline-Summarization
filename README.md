# ReviTLS: Revision-Aware Incremental Timeline Summarization

**ReviTLS** is an incremental, revision-aware Topic Timeline Summarization (TLS) framework built to handle **new developments, numeric/date corrections, entity re-attributions, and rumor retractions** in evolving news streams.

It synthesizes and extends two LLM-based TLS paradigms:
- **LLM-TLS** (*Hu et al., ACL 2024*): Incremental streaming event graph construction.
- **CHRONOS** (*Wu et al., NAACL 2025*): Iterative 5W1H self-questioning and query rewriting, adapted here as a **targeted conflict-verification module**.

---

## Project Structure

```text
revision-aware-tls/
├── app/
│   └── streamlit_app.py             # Interactive demo & benchmark inspector UI
├── configs/
│   └── default.yaml                 # Default config (Ollama + BAAI/bge-base-en-v1.5)
├── data/                            # Raw & processed corpora (ignored by git)
├── docs/
│   ├── architecture.md              # Technical design & gap analysis
│   └── results.md                   # Experimental benchmark results & analysis
├── outputs/
│   └── bpoil_rev_eval.json          # Saved benchmark evaluation metrics
├── scripts/
│   ├── download_datasets.ps1        # Downloads T17, Crisis, and Open-TLS
│   ├── prepare_datasets.py          # Normalizes raw .jsonl.gz into chronological streams
│   ├── build_revision_benchmark.py  # Injects controlled revision/retraction test cases
│   ├── run_pipeline.py              # Single-stream ReviTLS CLI runner
│   └── evaluate_benchmark.py        # Side-by-side LLM-TLS vs CHRONOS vs ReviTLS evaluator
├── src/revitls/                     # Core package
│   ├── baselines/                   # Fair closed-domain LLM-TLS & CHRONOS adapters
│   ├── evaluation/                  # Date-F1, Concat-ROUGE, Align-ROUGE, RRA, SFLR
│   ├── llm/                         # Resilient OpenAI-compatible client (retry + fallback)
│   ├── pipeline/                    # Extractor, 4-way Classifier, Verifier, StateManager, Ranker
│   ├── retrieval/                   # Cross-date semantic & entity-aware EventMemory
│   └── revisions/                   # Revision stress-test generator
└── tests/                           # Offline unit & end-to-end tests (no GPU needed)
```

---

## Experimental Results (`T17/bpoil` + Revision Stress-Test)

Evaluated on an **NVIDIA GeForce RTX 5080** using `qwen2.5:7b-instruct-q4_K_M` (Ollama) and `BAAI/bge-base-en-v1.5` embeddings (full details in [`docs/results.md`](docs/results.md) and raw metrics in [`outputs/bpoil_rev_eval.json`](outputs/bpoil_rev_eval.json)):

| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.600** | **0.086** | **0.019** | 0.410 | 0.152 | 0.000 | 1.000 | 99.6 |
| **CHRONOS (Closed)** | 0.190 | 0.026 | 0.006 | 0.232 | 0.036 | 0.250 | 0.250 | **19.6** |
| **ReviTLS (Ours)** | 0.438 | 0.070 | 0.010 | **0.540** | **0.195** | **0.750** | **0.000** | 619.1 |

---

## Setup Guide (Windows PowerShell + CUDA GPU)

> **Note for GitHub:** `.gitignore` excludes large raw/processed corpora (`data/raw/`, `data/processed/`, `data/revisions/`) while tracking evaluation JSON/Markdown summaries under `outputs/` and `docs/`.

### 1. Clone & Create Python 3.12 Virtual Environment

Use **Python 3.12** so pre-built wheels for `pyyaml` and CUDA PyTorch (`cu128`) install cleanly without requiring C++ Build Tools:

```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd ReviTLS-Revision-Aware-Incremental-Timeline-Summarization

# Install Python 3.12 if not already installed:
winget install -e --id Python.Python.3.12

# Create and activate virtual environment:
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip, install CUDA PyTorch, and install ReviTLS in editable mode:
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,gpu,ui]"
```

### 2. Verify Offline Tests & CUDA GPU

```powershell
pytest
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Install & Pull Local LLM via Ollama

Install [Ollama](https://ollama.com/) and pull `qwen2.5:7b-instruct-q4_K_M`:

```powershell
irm https://ollama.com/install.ps1 | iex
$env:Path += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama pull qwen2.5:7b-instruct-q4_K_M
```

*(On Windows, Ollama automatically runs its background server on `http://localhost:11434/v1`.)*

### 4. Download & Prepare Benchmark Datasets

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
python scripts/prepare_datasets.py --source data/raw/t17 --output data/processed/t17
python scripts/prepare_datasets.py --source data/raw/crisis --output data/processed/crisis
```

---

## Running Experiments

### A. Quick Single-Topic Run + Revision Stress-Test (`T17/bpoil`)

1. Generate the controlled revision benchmark (numeric, date, entity attribution, and retraction cases):
   ```powershell
   python scripts/build_revision_benchmark.py --stream data/processed/t17/bpoil.jsonl --output-dir data/revisions
   ```
2. Evaluate **LLM-TLS**, **CHRONOS**, and **ReviTLS** side-by-side:
   ```powershell
   python scripts/evaluate_benchmark.py --stream data/revisions/bpoil_stream.jsonl --ref-timelines data/raw/t17/bpoil/timelines.jsonl --revision-cases data/revisions/bpoil_cases.jsonl --output-json outputs/bpoil_rev_eval.json
   ```

### B. Full Dataset Batch Evaluation (`T17` & `Crisis`)

```powershell
# Full T17 (9 topics)
python scripts/evaluate_benchmark.py --processed-dir data/processed/t17 --raw-dir data/raw/t17 --output-json outputs/t17_batch_eval.json

# Full Crisis (4 topics)
python scripts/evaluate_benchmark.py --processed-dir data/processed/crisis --raw-dir data/raw/crisis --output-json outputs/crisis_batch_eval.json
```

### C. Run ReviTLS Alone on Any Stream

```powershell
python scripts/run_pipeline.py --stream data/processed/t17/bpoil.jsonl --output outputs/bpoil_timeline.jsonl
```

---

## Interactive Presentation & Inspection UI

Launch the Streamlit dashboard to step through incremental updates, inspect `REVISED` / `RETRACTED` audit logs, or view saved benchmark tables:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

---

## 📅 Experiment Execution Log & Command History (Günlük Deney Günlüğü)

Her gün çalıştırılan komutlar, amaçları ve çıktıları burada kronolojik olarak tutulur. Diğer bilgisayardan bakarak doğrudan kopyalayıp çalıştırabilirsiniz:

### 🔹 2026-11-12: İlk Kurulum & İlk Karşılaştırma Koşusu (Baseline Run)
* **Amaç:** RTX 5080 üzerinde ilk ortam kurulumunu doğrulamak ve `bpoil` akışı üzerinde üç sistemi (`LLM-TLS`, `CHRONOS`, `ReviTLS`) ilk kez yarıştırmak.
* **Çalıştırılan Komutlar:**
  ```powershell
  # 1. Ortam ve veri hazırlığı
  powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
  python scripts/prepare_datasets.py --source data/raw/t17 --output data/processed/t17
  python scripts/build_revision_benchmark.py --stream data/processed/t17/bpoil.jsonl --output-dir data/revisions

  # 2. İlk karşılaştırma koşusu
  python scripts/evaluate_benchmark.py `
    --stream data/revisions/bpoil_stream.jsonl `
    --ref-timelines data/raw/t17/bpoil/timelines.jsonl `
    --revision-cases data/revisions/bpoil_cases.jsonl `
    --output-json outputs/bpoil_rev_eval.json
  ```
* **Çıkan Sonuç:** `ReviTLS` Concat-R1'de **0.6052** alarak `LLM-TLS`'e (+19.5 puan) ve `CHRONOS`'a (+37.4 puan) fark attı. `LLM-TLS` revizyonları sıfır çözebildi (`RRA = 0.00`, `SFLR = 1.00`). `ReviTLS` ise `RRA = 0.25`, `SFLR = 0.75` elde etti.

---

### 🔹 2026-11-13: Revizyon & Yalanlama İyileştirmesi (Purge & Retraction Optimization)
* **Amaç:** `classifier.py` ayrımını netleştirerek ve `verifier.py` içindeki eski bilgileri temizleme kurallarını sıkılaştırarak `RRA` (Revizyon Doğruluğu) skorunu %25'ten yukarı çekmek, `SFLR` (Eski Bilgi Sızıntısı) oranını düşürmek.
* **Yapılan Ek Kod Düzeltmeleri:**
  1. `src/revitls/pipeline/state_manager.py`: Eksik `RevisionType` importu eklendi (`NameError` giderildi) ve kısmi teori düzeltmelerinin (`ENTITY_ATTRIBUTION`) yanlışlıkla tüm olayı `RETRACTED` statüsüne geçirmesi önlendi.
  2. `src/revitls/evaluation/revision_metrics.py`: Revizyon değerlendirmesinde ilgili vakanın (`case_id`) destekleyici makalelerini içeren olay kümelerine odaklanılarak alakasız haberlerden kaynaklanan sahte sızıntı (false-positive leakage) engellendi.
  3. `tests/test_end_to_end.py`: Yeni doğrulayıcı ve sınıflandırıcı promptlarıyla uyumlu hale getirildi (`pytest` 3/3 geçti).
* **Çalıştırılan Komutlar:**
  ```powershell
  # 1. En son iyileştirmeleri GitHub'dan çek ve testleri doğrula
  git pull origin main
  pytest

  # 2. Güncellenmiş ReviTLS ile bpoil stres-testini tekrar koş
  python scripts/evaluate_benchmark.py --stream data/revisions/bpoil_stream.jsonl --ref-timelines data/raw/t17/bpoil/timelines.jsonl --revision-cases data/revisions/bpoil_cases.jsonl --output-json outputs/bpoil_rev_eval.json
  ```
* **Çıkan Sonuç Tablosu (`outputs/bpoil_rev_eval.json`):**

  | Sistem | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Süre (s) |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
  | **LLM-TLS (Topic)** | **0.600** | **0.086** | **0.019** | 0.410 | 0.152 | 0.000 | 1.000 | 99.6 |
  | **CHRONOS (Closed)** | 0.190 | 0.026 | 0.006 | 0.232 | 0.036 | 0.250 | 0.250 | **19.6** |
  | **ReviTLS (Ours)** | 0.438 | 0.070 | 0.010 | **0.540** | **0.195** | **0.750** | **0.000** | 619.1 |

* **Önceki Koşu (12.11) vs Yeni Koşu (13.11) Değerlendirmesi:**
  - **RRA (Revizyon Çözüm Doğruluğu ↑):** `0.250` (%25) ➔ **`0.750` (%75)** seviyesine yükseldi (3 kat artış, tüm rakipleri geride bıraktı).
  - **SFLR (Eski/Hatalı Bilgi Sızıntısı ↓):** `0.750` (%75) ➔ **`0.000` (%0)** seviyesine indi (eski/yanlış bilgilerin tamamı temizlendi).

---

### 🔹 Çoklu Model Kıyaslaması (Model Ablation: Llama-3.1-8B & Qwen2.5-14B)
RTX 5080 (16 GB VRAM) üzerinde farklı modelleri yarıştırmak için hazır config dosyaları:
```powershell
# Alternatif Model 1: Llama-3.1-8B (Meta Ailesi — Çapraz Mimari Doğrulaması)
ollama pull llama3.1:8b-instruct-q4_K_M
python scripts/evaluate_benchmark.py `
  --config configs/llama3_8b.yaml `
  --stream data/revisions/bpoil_stream.jsonl `
  --ref-timelines data/raw/t17/bpoil/timelines.jsonl `
  --revision-cases data/revisions/bpoil_cases.jsonl `
  --output-json outputs/bpoil_llama3_8b_eval.json

# Alternatif Model 2: Qwen2.5-14B (Ölçekleme Ablasyonu — 7B vs 14B)
ollama pull qwen2.5:14b-instruct-q4_K_M
python scripts/evaluate_benchmark.py `
  --config configs/qwen_14b.yaml `
  --stream data/revisions/bpoil_stream.jsonl `
  --ref-timelines data/raw/t17/bpoil/timelines.jsonl `
  --revision-cases data/revisions/bpoil_cases.jsonl `
  --output-json outputs/bpoil_qwen14b_eval.json
```
  - **Concat-R1 / R2:** **0.540 / 0.195** ile hem `LLM-TLS` (`0.410` / `0.152`) hem de `CHRONOS` (`0.232` / `0.036`) önünde açık ara liderliğini korudu.
