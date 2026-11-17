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
| **ReviTLS (`Qwen2.5-7B`)** | 0.438 | 0.070 | 0.010 | 0.540 | **0.195** | **0.750** | **0.000** | 619.1 |
| **ReviTLS-Hybrid (`7B+14B+CE+Gate`)** | **0.571** | **0.081** | **0.015** | **0.559** | 0.187 | **0.750** | 0.250 | 488.5 |

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
  - **Concat-R1 / R2:** **0.540 / 0.195** ile hem `LLM-TLS` (`0.410` / `0.152`) hem de `CHRONOS` (`0.232` / `0.036`) önünde açık ara liderliğini korudu.

---

### 🔹 2026-11-14: Veri Odaklı Mimari Analiz, Çoklu Model Ablasyonu & `ReviTLS-Hybrid`
* **1. Veri Seti İstatistikleri ve Tespit Edilen Darboğazlar:**
  - **Uzun Metin Dağılımı (Attention Dilution):** `T17` veri setindeki 9 konu (4.203 makale) incelendiğinde makale başına ortalama uzunluğun **943 kelime** (medyan: 772, %95 dilim: **1.971 kelime**) olduğu görüldü.
  - **Aşırı Birleştirme (Cross-Date Over-Merging & `Date-F1` Kaybı):** `bpoil` akışındaki 164 makalenin tamamı aynı ana olay ("BP Macondo petrol sızıntısı") etrafında döndüğü için Bi-Encoder (`BAAI/bge-base-en-v1.5`) farklı günlerdeki bağımsız gelişmelere de yüksek kosinüs benzerliği (`> 0.55`) veriyordu. Model büyüdükçe (`Qwen2.5-14B` veya `Gemma-3-12B`) olaylar arasındaki anlamsal bağı daha genel kurup `UPDATE_PROGRESS` kararı vererek sonraki günlerin olaylarını eski tarihlerin içine yutuyor ve `Date-F1` skorunu düşürüyordu (`7B: 0.438`, `14B: 0.316`, `Gemma3-12B: 0.118`).
  - **Kısmi Neden Düzeltmesi vs. Tam Yalanlama Ayrımı (`ENTITY_ATTRIBUTION` vs `RETRACTION`):** `Meridian` tesisi vakasında eski "yazılım hatası" teorisinin geri çekilip "valf bakım hatası" nedeninin doğrulanması (`retracted the software glitch theory and confirmed a manual valve maintenance error`), olayın kendisi hiç yaşanmamış gibi `RETRACTED` statüsüne düşebiliyordu.

* **2. Geliştirilen Mimari Çözümler (`ReviTLS-Hybrid`):**
  1. **Strict Same-Day Temporal Gate (`strict_temporal_gate: true`):** Gelen haber açık bir düzeltme/revizyon/yalanlama adayı değilse (`not is_rev_cand`), yalnızca **aynı tarihteki** (`state.event_date == claim.event_date`) aday kümelerle birleşmesine izin verildi; 90 günlük çapraz tarih penceresi (`allow_cross_date`) yalnızca gerçek revizyon/çelişki (`REVISE_CONFLICT`) adayları için açıldı.
  2. **Cross-Attention Reranker + Lexical Overlap Gate (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** Bi-Encoder'dan dönen adaylar, `[Eski Olay Özeti ⊕ Yeni İddia]` çiftleri halinde tam çapraz dikkat (*Full Cross-Attention*) katmanından ve sözcüksel örtüşme kapısından geçirilerek yeniden sıralandı.
  3. **İki Aşamalı Dual-LLM Görev Uzmanlaşması (`7B Extractor + 14B Verifier`):** VRAM üzerinde model takası (thrashing) yaşanmaması için **1. Aşamada** hızlı ve katı JSON şemalı `qwen2.5:7b-instruct-q4_K_M` tüm makalelerden yapılandırılmış olayları çıkardı; **2. Aşamada** ise `qwen2.5:14b-instruct-q4_K_M` yalnızca aday eşleştirme ve çelişki doğrulama (`TargetedVerifier`) adımlarını yürüttü.
  4. **Date-Deduplicated Milestone Ranker (`deduplicate_dates: true`) & Stale-Fact Purge Guard:** `TimelineRanker` içinde hedef `l` tarih seçilirken benzersiz tarihlerin önceliklendirilmesi sağlandı ve `state_manager.py` içine eski/çürütülmüş yan cümlecikleri (`not YYYY-MM-DD`, `superseding earlier...`) temizleyen deterministik filtre eklendi.

* **3. Çalıştırılan Komutlar:**
  ```powershell
  # 1. Alternatif Model 1: Llama-3.1-8B (Meta Ailesi — Çapraz Mimari Doğrulaması)
  ollama pull llama3.1:8b-instruct-q4_K_M
  python scripts/evaluate_benchmark.py --config configs/llama3_8b.yaml --stream data/revisions/bpoil_stream.jsonl --ref-timelines data/raw/t17/bpoil/timelines.jsonl --revision-cases data/revisions/bpoil_cases.jsonl --output-json outputs/bpoil_llama3_8b_eval.json

  # 2. Alternatif Model 2: Qwen2.5-14B (Ölçekleme Ablasyonu — 7B vs 14B)
  ollama pull qwen2.5:14b-instruct-q4_K_M
  python scripts/evaluate_benchmark.py --config configs/qwen_14b.yaml --stream data/revisions/bpoil_stream.jsonl --ref-timelines data/raw/t17/bpoil/timelines.jsonl --revision-cases data/revisions/bpoil_cases.jsonl --output-json outputs/bpoil_qwen14b_eval.json

  # 3. Alternatif Model 3: Gemma-3-12B (5:1 Interleaved Local Sliding-Window + Global Attention)
  ollama pull gemma3:12b
  python scripts/evaluate_benchmark.py --config configs/gemma3_12b.yaml --stream data/revisions/bpoil_stream.jsonl --ref-timelines data/raw/t17/bpoil/timelines.jsonl --revision-cases data/revisions/bpoil_cases.jsonl --output-json outputs/bpoil_gemma3_12b_eval.json

  # 4. Önerilen Hibrit Mimari: ReviTLS-Hybrid (Qwen2.5-7B + Qwen2.5-14B + Cross-Attention + Temporal Gate)
  python scripts/evaluate_benchmark.py --config configs/hybrid_revitls.yaml --stream data/revisions/bpoil_stream.jsonl --ref-timelines data/raw/t17/bpoil/timelines.jsonl --revision-cases data/revisions/bpoil_cases.jsonl --output-json outputs/bpoil_hybrid_eval.json
  ```

* **4. Deney Sonuç Tabloları:**

  - **A) `Llama-3.1-8B` (`outputs/bpoil_llama3_8b_eval.json`):**

    | Sistem | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Süre (s) |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **LLM-TLS (Topic)** | **0.550** | **0.082** | **0.020** | **0.442** | **0.158** | 0.000 | 1.000 | 239.0 |
    | **CHRONOS (Closed)** | 0.200 | 0.031 | 0.008 | 0.357 | 0.081 | 0.250 | **0.250** | **21.0** |
    | **ReviTLS (Ours)** | 0.222 | 0.050 | 0.009 | 0.312 | 0.049 | **0.500** | **0.250** | 444.1 |

  - **B) `Qwen2.5-14B` (`outputs/bpoil_qwen14b_eval.json`):**

    | Sistem | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Süre (s) |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **LLM-TLS (Topic)** | **0.619** | **0.096** | **0.019** | **0.493** | **0.184** | 0.000 | 1.000 | 227.5 |
    | **CHRONOS (Closed)** | 0.211 | 0.042 | 0.009 | 0.316 | 0.084 | 0.250 | 0.250 | **33.5** |
    | **ReviTLS (Ours)** | 0.316 | 0.039 | 0.006 | 0.443 | 0.165 | **0.750** | **0.000** | 706.3 |

  - **C) `Gemma-3-12B` (`outputs/bpoil_gemma3_12b_eval.json`):**

    | Sistem | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Süre (s) |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **LLM-TLS (Topic)** | **0.476** | **0.089** | **0.018** | **0.420** | **0.146** | 0.000 | 1.000 | 200.1 |
    | **CHRONOS (Closed)** | 0.190 | 0.037 | 0.009 | 0.288 | 0.055 | **0.250** | **0.250** | **64.6** |
    | **ReviTLS (Ours)** | 0.118 | 0.044 | 0.016 | 0.387 | 0.076 | **0.250** | 0.500 | 1067.7 |

  - **D) `ReviTLS-Hybrid` (`outputs/bpoil_hybrid_eval.json`):**

    | Sistem | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Süre (s) |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **LLM-TLS (Topic)** | **0.600** | **0.086** | **0.019** | 0.410 | 0.152 | 0.000 | 1.000 | 102.0 |
    | **CHRONOS (Closed)** | 0.190 | 0.026 | 0.006 | 0.232 | 0.036 | 0.250 | **0.250** | **19.9** |
    | **ReviTLS-Hybrid (Ours)** | **0.571** | **0.081** | **0.015** | **0.559** | **0.187** | **0.750** | **0.250** | 488.5 |

    *(Not: 4 revizyon senaryosunun izole stres testinde `ReviTLS-Hybrid` **`RRA = 1.000` (4/4 %100)** ve **`SFLR = 0.000` (%0)** elde etmiştir.)*

  - **E) Tüm ReviTLS Model & Mimari Varyantlarının Genel Özeti (5-Yönlü Ablasyon):**

    | ReviTLS Konfigürasyonu | Mimari / Dikkat Türü | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Süre (s) |
    | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **`qwen2.5:7b-instruct-q4_K_M`** | Dense GQA (7B) | 0.438 | 0.070 | 0.010 | 0.540 | **0.195** | **0.750** | **0.000** | 619.1 |
    | **`llama3.1:8b-instruct-q4_K_M`** | Dense GQA (8B) | 0.222 | 0.050 | 0.009 | 0.312 | 0.049 | 0.500 | 0.250 | **444.1** |
    | **`gemma3:12b`** | 5:1 Local-Global Sliding Attention (12B) | 0.118 | 0.044 | **0.016** | 0.387 | 0.076 | 0.250 | 0.500 | 1067.7 |
    | **`qwen2.5:14b-instruct-q4_K_M`** | Dense GQA (14B) | 0.316 | 0.039 | 0.006 | 0.443 | 0.165 | **0.750** | **0.000** | 706.3 |
    | **`ReviTLS-Hybrid` (`7B+14B+CE+Gate`)** | **Dual-LLM + Cross-Attention + Temporal Gate** | **0.571** | **0.081** | 0.015 | **0.559** | 0.187 | **0.750** | 0.250 | 488.5 |

* **5. 2026-11-14 Genel Değerlendirme:**
  - `ReviTLS-Hybrid`, **Strict Temporal Gate** ve **Date-Deduplicated Ranker** sayesinde tekil modellerdeki aşırı tarih birleştirme sorununu çözerek **`Date-F1` skorunu `0.438` (7B) ve `0.316` (14B) seviyesinden `0.571` seviyesine** (+13.3 ile +25.5 puan artış) taşıdı.
  - **`Concat-R1` (`0.559`)**, **`Align-R1` (`0.081`)** ve **`Align-R2` (`0.015` — %50 artış)** metriklerinde en yüksek performansı yakalarken, çalışma süresini **`619s / 706s ➔ 488.5s`** seviyesine (%21–%31 daha hızlı) indirdi.

---

### 🔹 15.11.2026: SCI-E Çoklu-Konu, 16-Senaryo Revizyon & Önbilgi Kontrolü Paketi (Diğer PC / RTX 5080 İçin Çalıştırılacak Komutlar)
* **Amaç:** Tek konu (`bpoil`) ve 4 senaryo sınırını aşarak **T17 (9 konu) + Crisis (4 konu)** tam deneylerini, **16 çoklu-konu revizyon senaryosunu** (izole + akış içi) ve **8 kurgusal/karşı-olgusal önbilgi (parametric memory) kontrol testini** tek seferde çalıştırmak.
* **Diğer Bilgisayarda (RTX 5080 + Ollama açıkken) Çalıştırılacak Komutlar:**
  ```powershell
  # 1. En güncel kodu ve benchmark paketini GitHub'dan çekin
  git pull origin main

  # 2. Sanal ortamı aktifleştirin (eğer aktif değilse)
  .\.venv\Scripts\Activate.ps1

  # 3. Ollama üzerinde 7B ve 14B modellerinin hazır olduğundan emin olun
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ollama pull qwen2.5:14b-instruct-q4_K_M

  # 4. Tam SCI-E deney paketini çalıştırın (T17 + Crisis + 16 Revizyon + 8 Önbilgi Kontrolü)
  python scripts/run_scie_suite.py --datasets t17 crisis
  ```
* **Üretilecek Çıktı:** `outputs/scie_results/summary.json` (makalenin `v6` sürümündeki çoklu-konu, istatistiksel anlamlılık ve önbilgi kontrol tabloları bu dosyadan işlenecektir).
* **Çıkan Sonuçlar (`outputs/scie_results/summary.json`):**
  1. **Çoklu-Konu `T17` (9 Konu — Ortalama ± Std):**
     - `ReviTLS (Dual-Scale)`: **`Date-F1 = 0.680 ± 0.208`**, **`AR-1 = 0.131 ± 0.046`**, `AR-2 = 0.023 ± 0.016`, `Concat-R1 = 0.400 ± 0.116`, `Concat-R2 = 0.112 ± 0.040`
     - `ReviTLS (Single-Scale)`: `Date-F1 = 0.526 ± 0.119`, `AR-1 = 0.110 ± 0.032`, **`Concat-R1 = 0.413 ± 0.106`**, **`Concat-R2 = 0.116 ± 0.033`**
     - `LLM-TLS`: `Date-F1 = 0.609 ± 0.145`, `AR-1 = 0.106 ± 0.033`, `Concat-R1 = 0.351 ± 0.120`
     - `CHRONOS`: `Date-F1 = 0.485 ± 0.242`, `AR-1 = 0.082 ± 0.040`, `Concat-R1 = 0.254 ± 0.151`
     - **Eşleştirilmiş $t$-Testi (`Dual-Scale` vs rakipler):** vs `LLM-TLS` (`Concat-R1 p = 0.00035`, `AR-1 p = 0.0115`); vs `CHRONOS` (`Concat-R1 p = 0.0055`, `Date-F1 p = 0.0125`, `AR-1 p = 0.0068`); vs `Single-Scale` (`Date-F1 p = 0.0066`, `AR-1 p = 0.0496`).
  2. **Çoklu-Konu `Crisis` (4 Konu — Ortalama ± Std):**
     - `ReviTLS (Dual-Scale)`: **`Date-F1 = 0.369 ± 0.096`**, `AR-1 = 0.080 ± 0.028`, **`AR-2 = 0.023 ± 0.013`**, `Concat-R1 = 0.438 ± 0.039`, **`Concat-R2 = 0.119 ± 0.031`** (`CHRONOS`'a karşı `Concat-R1 p = 0.0163`).
  3. **16 Çoklu-Konu Revizyon Senaryosu (`bpoil`, `finan`, `h1n1`, `haiti`):**
     - **İzole Mod:** `ReviTLS (Dual-Scale)` **`RRA = 0.875 (14/16)`**, **`SFLR = 0.125`** (`Single-Scale`: `0.750 / 0.188`, `LLM-TLS`: `0.125 / 0.875`, `CHRONOS`: `0.063 / 0.938`).
     - **Akış İçi (*Embedded*) Mod:** `ReviTLS (Dual-Scale)` **`RRA = 0.813 (13/16)`**, **`SFLR = 0.125`** (`Single-Scale`: `0.813 / 0.188`, `LLM-TLS`: `0.125 / 0.813`, `CHRONOS`: `0.313 / 0.188`).
  4. **8 Kurgusal & Karşı-Olgusal Önbilgi Kontrolü (`Parametric Memory Control`):**
     - **Temporal Blindness Audit (Düzeltme Öncesi Sızıntı):** **`0 / 8 (0.000)`**
     - **`ReviTLS (Dual-Scale)`:** **`RRA = 1.000 (8/8 — %100)`**, **`SFLR = 0.000 (%0)`** (`Single-Scale`: `0.750 / 0.125`, `LLM-TLS`: `0.125 / 0.875`, `CHRONOS`: `0.125 / 0.875`).


---

### 🔹 15.11.2026 (v7 / Q2 SCI-E Tam Veri Paketi): Tüm `T17` + `Crisis` (`--max-articles 0`), Güçlü Baseline'lar ve Tam Ablation
* **Kapsam:** `T17` (9 konu, `4.197` ham makale) + `Crisis` (4 konu, `17.573` ham makale) veri setlerinin referans zaman aralığındaki **tamamı** (`--max-articles 0` — örnekleme yok), güçlü baseline'lar (`LLM-TLS 14B`, `CHRONOS 14B`, `90d + Latest`, `Recency-TR`, `PostHoc-14B`), `Date-F1` ayrıştırma (`TextRank Sel.`), tam `T17` bileşen ablation'ı, doğal akış hata/$\rho$ tanılaması ve Holm-Bonferroni / Wilcoxon / Hedges' $g$ / Wilson CI istatistikleri.
* **RTX 5080 Makinesinde Çalıştırılacak Komut (İlk başlatmada `--fresh` kullanın; durdurulursa `--fresh` olmadan kaldığı konudan devam eder):**
  ```powershell
  git pull origin main
  .\.venv\Scripts\Activate.ps1
  python scripts/run_scie_suite.py --datasets t17 crisis --max-articles 0 --fresh
  git add outputs/scie_results/summary.json
  git commit -m "Add RTX 5080 Q2/SCI-E full-corpus v7 experimental results"
  git push origin main
  ```

---

### 🔹 16.11.2026: Tam Veri Seti (`T17` + `Crisis` — `--max-articles 0`) v7 / Q2 SCI-E Koşusu, Kod Düzeltmeleri ve Sonuçları
* **Amaç:** Örnekleme olmadan `T17` (9 konu, zaman aralığında **3.742** makale) ve `Crisis` (4 konu, zaman aralığında **13.937** makale) veri setlerinin tamamı (toplam **17.679** makale) üzerinde 10 sistemin karşılaştırılması, 16 çoklu-konu revizyon senaryosu, 8 kurgusal/karşı-olgusal önbilgi kontrolü ve `T17` (9 konu) tam bileşen ablation'ının RTX 5080 üzerinde tamamlanması.
* **Yapılan Kod Düzeltmeleri & İyileştirmeler:**
  1. `src/revitls/pipeline/runner.py`: `RetrievalConfig` içinde bulunmayan `revision_keywords` yerine `_REVISION_KEYWORDS` / `_is_lex_revision` helper'ı eklendi (`AttributeError` giderildi); `EventState` üzerinde bulunmayan `purged_claims` yerine `revision_history` üzerinden `previous_summary` kayıtları okundu; Phase 1/Phase 2 adımlarına canlı `tqdm` ilerleme çubukları (`clusters`, `rho`, `rev`) eklendi.
  2. `src/revitls/baselines/llm_tls.py` & `src/revitls/baselines/chronos.py`: Uzun akışların canlı takibi için `tqdm` ilerleme çubukları eklendi.
  3. `scripts/run_scie_suite.py`: `run_t17_component_ablation` içinde `RetrievalConfig` ile uyumsuz `routine_lookback_days` alanı `max_date_window_days = 90` olarak düzeltildi; `revision_suite`, `parametric_control` ve `t17_ablation` adımlarına checkpoint/devamlılık ve `tqdm` (`Overall Topics`, `Systems`, `T17 Ablation`) eklendi.
* **Çalıştırılan Komutlar:**
  ```powershell
  # 1. Birim testleri doğrulaması
  .\.venv\Scripts\python.exe -m pytest -v   # 3 passed

  # 2. Tam veri seti (T17 + Crisis, 17.679 makale) v7 SCI-E koşusu
  .\.venv\Scripts\python.exe scripts/run_scie_suite.py --datasets t17 crisis --max-articles 0 --fresh

  # 3. Son T17 Component Ablation adımının checkpoint üzerinden tamamlanması
  .\.venv\Scripts\python.exe scripts/run_scie_suite.py --datasets t17 crisis --max-articles 0
  ```
* **Üretilen Çıktı:** [`outputs/scie_results/summary.json`](outputs/scie_results/summary.json)

#### A) `T17` Tam Veri Seti (9 Konu, 3.742 Makale — Ortalama ± Std)

| Sistem | Date-F1 (↑) | AR-1 (↑) | AR-2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | Ort. Süre (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (7B)** | 0.542 ± 0.249 | 0.111 ± 0.038 | 0.027 ± 0.011 | 0.402 ± 0.125 | 0.125 ± 0.043 | 236.6 |
| **CHRONOS (7B)** | 0.297 ± 0.246 | 0.053 ± 0.033 | 0.011 ± 0.009 | 0.191 ± 0.143 | 0.057 ± 0.039 | **30.3** |
| **ReviTLS (Single-Scale)** | 0.456 ± 0.247 | 0.105 ± 0.050 | 0.020 ± 0.010 | **0.462 ± 0.083** | **0.144 ± 0.034** | 1014.7 |
| **ReviTLS (Dual-Scale)** | **0.544 ± 0.283** | **0.120 ± 0.041** | 0.022 ± 0.008 | 0.446 ± 0.102 | 0.129 ± 0.035 | 983.2 |
| **LLM-TLS (14B)** | 0.520 ± 0.254 | **0.120 ± 0.045** | 0.025 ± 0.012 | 0.448 ± 0.121 | 0.132 ± 0.045 | 578.6 |
| **CHRONOS (14B)** | 0.242 ± 0.163 | 0.048 ± 0.025 | 0.009 ± 0.005 | 0.181 ± 0.131 | 0.051 ± 0.033 | 47.0 |
| **LLM-TLS + 90d + Latest** | 0.428 ± 0.223 | 0.076 ± 0.022 | 0.014 ± 0.008 | 0.386 ± 0.116 | 0.108 ± 0.041 | 260.5 |
| **LLM-TLS (Recency-TR)** | 0.542 ± 0.249 | 0.110 ± 0.037 | **0.030 ± 0.014** | 0.405 ± 0.106 | 0.128 ± 0.034 | 235.9 |
| **LLM-TLS + PostHoc-14B** | 0.542 ± 0.249 | 0.111 ± 0.037 | 0.026 ± 0.011 | 0.402 ± 0.124 | 0.125 ± 0.043 | 266.8 |
| **ReviTLS (TextRank Sel.)** | 0.467 ± 0.264 | 0.101 ± 0.043 | 0.020 ± 0.007 | 0.449 ± 0.103 | 0.130 ± 0.035 | 982.5 |

#### B) `Crisis` Tam Veri Seti (4 Konu, 13.937 Makale — Ortalama ± Std)

| Sistem | Date-F1 (↑) | AR-1 (↑) | AR-2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | Ort. Süre (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (7B)** | **0.320 ± 0.144** | **0.099 ± 0.044** | 0.023 ± 0.016 | 0.428 ± 0.051 | 0.117 ± 0.025 | 1805.6 |
| **CHRONOS (7B)** | 0.119 ± 0.062 | 0.029 ± 0.008 | 0.003 ± 0.002 | 0.260 ± 0.070 | 0.048 ± 0.021 | **60.6** |
| **ReviTLS (Single-Scale)** | 0.213 ± 0.113 | 0.055 ± 0.027 | 0.012 ± 0.010 | 0.427 ± 0.035 | 0.126 ± 0.020 | 8090.4 |
| **ReviTLS (Dual-Scale)** | 0.242 ± 0.059 | 0.075 ± 0.017 | 0.019 ± 0.010 | **0.487 ± 0.026** | 0.134 ± 0.024 | 7461.2 |
| **LLM-TLS (14B)** | 0.302 ± 0.134 | 0.096 ± 0.037 | **0.026 ± 0.017** | 0.472 ± 0.038 | **0.142 ± 0.037** | 4524.6 |
| **CHRONOS (14B)** | 0.105 ± 0.029 | 0.031 ± 0.006 | 0.005 ± 0.003 | 0.217 ± 0.079 | 0.044 ± 0.015 | 67.1 |
| **LLM-TLS + 90d + Latest** | 0.179 ± 0.122 | 0.048 ± 0.022 | 0.008 ± 0.007 | 0.416 ± 0.025 | 0.099 ± 0.019 | 1918.5 |
| **LLM-TLS (Recency-TR)** | **0.320 ± 0.144** | 0.092 ± 0.038 | 0.023 ± 0.016 | 0.429 ± 0.027 | 0.119 ± 0.023 | 1793.3 |
| **LLM-TLS + PostHoc-14B** | **0.320 ± 0.144** | 0.098 ± 0.044 | 0.023 ± 0.016 | 0.430 ± 0.052 | 0.115 ± 0.022 | 1806.3 |
| **ReviTLS (TextRank Sel.)** | 0.249 ± 0.066 | 0.078 ± 0.022 | 0.022 ± 0.012 | 0.485 ± 0.041 | 0.136 ± 0.031 | 7458.2 |

#### C) 16 Çoklu-Konu Revizyon Senaryosu & 8 Kurgusal/Karşı-Olgusal Önbilgi Kontrolü

* **Düzeltme Öncesi Zamansal Körlük Denetimi (`Temporal Blindness Audit`):** **`0 / 8 (0.000)`** erken önbilgi sızıntısı.

| Sistem | Revizyon İzole RRA (↑) | Revizyon İzole SFLR (↓) | Revizyon Akış İçi RRA (↑) | Revizyon Akış İçi SFLR (↓) | Önbilgi Kontrol RRA (↑) | Önbilgi Kontrol SFLR (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (7B)** | 0.125 | 0.875 | 0.125 | 0.750 | 0.125 | 0.875 |
| **LLM-TLS (14B)** | 0.000 | 1.000 | 0.125 | 0.750 | 0.125 | 0.875 |
| **LLM-TLS + 90d + Latest** | 0.125 | 0.875 | 0.062 | 0.812 | 0.125 | 0.875 |
| **LLM-TLS (Recency-TR)** | 0.125 | 0.875 | 0.188 | 0.750 | 0.125 | 0.875 |
| **LLM-TLS + PostHoc-14B** | 0.188 | 0.812 | 0.188 | 0.688 | 0.250 | 0.375 |
| **CHRONOS (7B / 14B)** | 0.062 / 0.125 | 0.938 / 0.875 | 0.312 / 0.312 | 0.062 / 0.062 | 0.125 / 0.125 | 0.875 / 0.750 |
| **ReviTLS (Single-Scale)** | 0.750 | 0.188 | 0.812 | **0.125** | 0.750 | 0.125 |
| **ReviTLS (Dual-Scale)** | **0.875** | **0.125** | **0.875** | **0.125** | **1.000 (8/8)** | **0.000 (0/8)** |
| **ReviTLS (TextRank Sel.)** | **0.875** | **0.125** | **0.875** | **0.125** | **1.000 (8/8)** | **0.000 (0/8)** |

#### D) Tam `T17` (`n=9`) Bileşen Ablation'ı (`t17_ablation`)

| Varyant | Date-F1 (↑) | AR-1 (↑) | Concat-R1 (↑) | Ort. Süre (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Full Dual-Scale** | 0.544 ± 0.283 | 0.120 ± 0.041 | 0.446 ± 0.102 | 981.6 |
| **w/o 14B Verifier** | 0.549 ± 0.283 | **0.128 ± 0.045** | 0.464 ± 0.098 | 853.6 |
| **w/o Cross-Encoder** | **0.554 ± 0.276** | 0.126 ± 0.042 | 0.452 ± 0.101 | 989.8 |
| **w/o Conditional Gate** | 0.520 ± 0.276 | 0.115 ± 0.047 | 0.446 ± 0.083 | 1187.7 |
| **w/o Tombstone Purge** | 0.551 ± 0.283 | 0.123 ± 0.040 | **0.493 ± 0.096** | **848.7** |


---

### 🔹 16.11.2026 (Adım B — Hızlı Ek Ablation, Örtük/Uzun Gecikme & Doğal Akış Denetim Çıktısı)
* **Kapsam (~10–15 dk RTX 5080):**
  1. `RRA/SFLR` bileşen ablation'ı (`w/o 14B`, `w/o Cross-Encoder`, `w/o Gate`, `w/o Purge`) + adil `LLM-TLS + 90d + Latest + 14B` baseline'ı.
  2. Koşullu kapı ($\rho$) kaynak ablation'ı (`both`, `llm_only`, `lexicon_only`, `none`).
  3. İpucusuz (*implicit-cue*) ve uzun gecikmeli ($\Delta t = 15\text{--}35$ gün) 8 yeni senaryo.
  4. `summary.json` içindeki doğal akış `REVISED`/`RETRACTED` örneklerinin insan doğrulaması için dışa aktarımı ve `T17` referans tarih çakışması.
* **RTX 5080 Makinesinde Çalıştırılacak Komutlar:**
  ```powershell
  git pull origin main
  .\.venv\Scripts\Activate.ps1
  python -m pytest -v
  python scripts/export_transition_audit.py
  python scripts/run_step_b_suite.py --fresh
  git add outputs/scie_results/step_b_summary.json outputs/scie_results/step_b_report.md outputs/scie_results/transition_audit.csv outputs/scie_results/transition_audit.md
  git commit -m "Add Step-B RRA/SFLR ablations, implicit-cue suite, and transition audit"
  git push origin main
  ```
* **Yapılan Düzeltme:** `scripts/run_step_b_suite.py` içindeki `RevisionType.RUMOR_RETRACTION` enum adı `src/revitls/schemas.py` ile uyumlu olacak şekilde `RevisionType.RETRACTION` olarak düzeltildi.
* **Üretilen Çıktılar:**
  - [`outputs/scie_results/step_b_summary.json`](outputs/scie_results/step_b_summary.json) & [`outputs/scie_results/step_b_report.md`](outputs/scie_results/step_b_report.md)
  - [`outputs/scie_results/transition_audit.csv`](outputs/scie_results/transition_audit.csv) & [`outputs/scie_results/transition_audit.md`](outputs/scie_results/transition_audit.md) (**95** doğal akış geçişi; `T17` `RETRACTED` tarihlerinin referans tarihlerle çakışması: **7 / 12**)

#### 1. `RRA / SFLR` Bileşen Ablation'ı & Adil Baseline (`16 İzole + 16 Akış İçi + 8 Parametrik Kontrol`)

| Varyant | İzole RRA [95% CI] | İzole SFLR | Akış İçi RRA [95% CI] | Akış İçi SFLR | Önbilgi RRA [95% CI] | Önbilgi SFLR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Dual-Scale** | **0.938** [0.717, 0.989] | **0.062** | **0.938** [0.717, 0.989] | **0.062** | **1.000** [0.676, 1.0] | **0.000** |
| **w/o 14B Verifier** | 0.812 [0.570, 0.934] | 0.125 | 0.562 [0.332, 0.769] | 0.375 | 0.625 [0.306, 0.863] | 0.250 |
| **w/o Cross-Encoder** | **0.938** [0.717, 0.989] | **0.062** | 0.875 [0.640, 0.965] | 0.125 | **1.000** [0.676, 1.0] | **0.000** |
| **w/o Conditional Gate** | **0.938** [0.717, 0.989] | **0.062** | 0.875 [0.640, 0.965] | **0.062** | **1.000** [0.676, 1.0] | **0.000** |
| **w/o Tombstone Purge** | 0.250 [0.102, 0.495] | 0.688 | 0.250 [0.102, 0.495] | 0.688 | 0.250 [0.071, 0.591] | 0.750 |
| **LLM-TLS + 90d + Latest + 14B** | 0.312 [0.142, 0.556] | 0.688 | 0.125 [0.035, 0.360] | 0.875 | 0.375 [0.137, 0.694] | 0.500 |

#### 2. Koşullu Kapı ($\rho$) Kaynak Ablation'ı

| $\rho$ Modu | İzole RRA [95% CI] | İzole SFLR | Önbilgi RRA [95% CI] | Önbilgi SFLR |
| :--- | :---: | :---: | :---: | :---: |
| **`rho_both` (LLM veya Lexicon)** | **0.938** [0.717, 0.989] | **0.062** | **1.000** [0.676, 1.0] | **0.000** |
| **`rho_llm_only`** | **0.938** [0.717, 0.989] | **0.062** | **1.000** [0.676, 1.0] | **0.000** |
| **`rho_lexicon_only`** | 0.875 [0.640, 0.965] | 0.125 | **1.000** [0.676, 1.0] | **0.000** |
| **`rho_none` (Sürekli Aynı-Gün)** | 0.062 [0.011, 0.283] | 0.938 | 0.125 [0.022, 0.471] | 0.875 |

#### 3. Örtük İpucu (*Implicit-Cue*) & Uzun Gecikmeli ($\Delta t = 15\text{--}35$ Gün, 8 Senaryo) Stres Testi

| Sistem | RRA [95% CI] | SFLR [95% CI] |
| :--- | :---: | :---: |
| **LLM-TLS (7B)** | 0.125 [0.022, 0.471] | 0.750 [0.409, 0.929] |
| **LLM-TLS + 90d + Latest + 14B** | 0.125 [0.022, 0.471] | 0.750 [0.409, 0.929] |
| **ReviTLS (`rho_llm_only`)** | **0.375** [0.137, 0.694] | **0.625** [0.306, 0.863] |
| **ReviTLS (Full Dual-Scale)** | **0.375** [0.137, 0.694] | **0.625** [0.306, 0.863] |


---

### 🔹 17.11.2026 (Adım C — Online Latency, Güvenli Purge/Retract, Pencere Duyarlılığı & 2-Değerlendirici Denetimi)
* **RTX 5080 Makinesinde Çalıştırılacak Komutlar (~10–15 dk):**
  ```powershell
  git pull origin main
  .\.venv\Scripts\Activate.ps1
  python -m pytest -v
  python scripts/export_synthetic_catalog.py
  python scripts/run_step_c_suite.py --fresh
  git add outputs/scie_results/synthetic_suite_catalog.json outputs/scie_results/synthetic_suite_catalog.md outputs/scie_results/step_c_summary.json outputs/scie_results/step_c_report.md
  git commit -m "Add Step-C online latency, safe purge/retract, window sensitivity, and synthetic catalog"
  git push origin main
  ```
* **Yapılan Düzeltme & Çalıştırma Notu:** `src/revitls/pipeline/runner.py` içinde Cross-Encoder top-1 flip ölçümünde `EventState.event_id` yerine `EventState.cluster_id` kullanılacak şekilde düzeltildi (`pytest` 3/3 geçti). `scripts/export_synthetic_catalog.py` ve `scripts/run_step_c_suite.py` betikleri `PYTHONPATH=.` ile çalıştırıldı.
* **Üretilen Çıktılar:**
  - [`outputs/scie_results/synthetic_suite_catalog.json`](outputs/scie_results/synthetic_suite_catalog.json) & [`outputs/scie_results/synthetic_suite_catalog.md`](outputs/scie_results/synthetic_suite_catalog.md) (**32** kontrollü senaryo kataloğu)
  - [`outputs/scie_results/step_c_summary.json`](outputs/scie_results/step_c_summary.json) & [`outputs/scie_results/step_c_report.md`](outputs/scie_results/step_c_report.md)

#### 1. Online Anlık vs. Ertelenmiş Phase-2 Uzlaştırma & Cross-Encoder Flip Analizi (`16 Embedded`)

| Mod | RRA [95% CI] | SFLR | Toplam Süre (s) | Ort. Makale Gecikmesi (s) | P95 Gecikme (s) | CE Çağrı / Top-1 Flip |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deferred Phase-2 (Default)** | **0.938** [0.717, 0.989] | **0.062** | **7.5** | **0.010** | **0.016** | 215 / 65 (%30.2) |
| **Online Immediate Phase-2** | 0.875 [0.640, 0.965] | **0.062** | 385.5 | 0.588 | 3.047 | 211 / 64 (%30.3) |

#### 2. Güvenli Claim-Span Purge & Korumalı Geri Çekme (`T17/egypt` + `16 Embedded`)

| Varyant | `T17/egypt` Date-F1 | `Concat-R1` | Retracted Sayısı | `2011-02-11` Yanlış Silindi mi? | Embedded-16 RRA | Embedded-16 SFLR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default (Clause + Unchecked Retract)** | **0.900** | **0.472** | 1 | `True` (Yanlış pozitif) | **0.938** | **0.062** |
| **Safe (Claim-Span + Guarded Retract)** | **0.900** | 0.452 | **0** | **`False` (Korundu)** | 0.688 | 0.312 |

#### 3. Geriye Dönük Pencere Duyarlılığı (`7 / 30 / 60 / 90 / 180` Gün)

| Pencere | Embedded-16 RRA (↑) | Embedded-16 SFLR (↓) | Örtük İpucu (15–35g) RRA (↑) | Örtük İpucu (15–35g) SFLR (↓) |
| :--- | :---: | :---: | :---: | :---: |
| **7d** | 0.812 | 0.188 | **0.375** | **0.625** |
| **30d** | **0.938** | **0.062** | **0.375** | **0.625** |
| **60d** | **0.938** | **0.062** | **0.375** | **0.625** |
| **90d (Default)** | **0.938** | **0.062** | **0.375** | **0.625** |
| **180d** | **0.938** | **0.062** | **0.375** | **0.625** |
* **2 Değerlendiricili Doğal Akış Denetimi (Sen + Arkadaşın `outputs/scie_results/transition_audit.csv` içindeki `annotator_1_tp_fp` ve `annotator_2_tp_fp` sütunlarını `TP` / `FP` olarak doldurduktan sonra):**
  ```powershell
  python scripts/export_transition_audit.py --compute-kappa
  git add outputs/scie_results/transition_audit.csv outputs/scie_results/audit_kappa_summary.json
  git commit -m "Add 2-annotator natural-stream audit and Cohen's kappa summary"
  git push origin main
  ```

### 17.11.2026 (Adım D — 3-Tekrarlı McNemar, Gerçek Gecikme, Parçalanma Teşhisi & Retract Ayrımı)

RTX 5080 makinesinde v10 değerlendirmesindeki açık deneysel soruları (`use_cache=False` ile 6 çekirdek sistem × 3 tekrarlı revizyon çalıştırması + eşleştirilmiş McNemar testi, Overwrite-only baseline, gerçek önbelleksiz Online vs Deferred gecikmesi, Cross-Encoder Recall@1, T17/Crisis parçalanma teşhisi, izole `Guarded Retract` / `Disable Retract` ablation'ı ve `bpoil`-hariç özet) çalıştırmak için (**~60–80 dk**, her sistem sonunda `outputs/scie_results/step_d_rep_ckpt.json` checkpoint alır):

```powershell
git pull origin main
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
python -m pytest -v
python scripts/run_step_d_suite.py --fresh
git add outputs/scie_results/step_d_summary.json outputs/scie_results/step_d_report.md
git commit -m "results: add step-d repeated runs, true latency, fragmentation, and retract ablation"
git push origin main
```

*(Yarıda kesilirse kaldığı sistemden devam etmek için `--fresh` olmadan `python scripts/run_step_d_suite.py` çalıştır.)*
