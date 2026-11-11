# Hebrew Text Readability & Autonomous Simplification (`heb-text-lvl`)

Production machine-learning infrastructure for **Hebrew text readability assessment** and **closed-loop agentic text simplification** across 14 K-12, L2 Ulpan/Hadshon, Hearing-Impaired, Easy-Read (`intellectual_disability`), and Academic/Technical reading levels.

---

## Architecture Overview (v3)

1. **Layer 1 — Centralized Configuration (`src/hebrew_readability/config.py`, `src/hebrew_readability/hf_hub.py`):**
   - 11 purged linguistic features (`FINAL_FEATURE_ORDER`), per-level `Mean_Abs_SHAP` weights (`LEVEL_SHAP_WEIGHTS`), and per-raw-level transformed feature medians (`RAW_LEVEL_MEDIANS`).
   - Calibration Dead-Zone Bridging (`[3.2720, 3.3898]` between Grade 2 and Grade 3), empirical anchor maps (`build_empirical_level_anchor_map`), and scale-normalized IQR distance ($d_{\text{IQR}}$).
   - Continuous inverse-median vocalization targets (`CONTINUOUS_VOCALIZATION_CONFIG`) and sub-paragraph splitting policies (`SUB_PARAGRAPH_SPLIT_CONFIG`).
2. **Layer 2 — Stateful Inference & Attribution Engine (`src/inference/engine.py`):**
   - Singleton loading of Stanza (`he`), DictaBERT (`dicta-il/dictabert-joint`), `dicta-1.0.int8.onnx`, cascading lemma-POS frequency dictionaries, fitted `HebrewReadabilityPreprocessor`, and the frozen XGBoost readability regressor.
   - Option A Windowed Volume Normalization (`_apply_windowed_volume_normalization`), contextual Nikkud damping, sub-paragraph chunk scoring (`\n\n`), track-separated Z-score probabilities (K-12 vs. Special/L2), C++ TreeSHAP attributions, and DictaBERT-driven `dicta-onnx` vocalization.
3. **Layer 3 — Autonomous Simplification Agent (`src/inference/agent.py`):**
   - Closed-loop self-correction (`gpt-4o`) using exact raw-level Core IQR verification ($Q_1 \le \hat{Y} \le Q_3$), Hybrid Differential TreeSHAP feedback with draft-specific evidence (longest sentence & lowest-Zipf lemmas), continuous `dicta-onnx` vocalization plans, and a Dynamic Stagnation Breaker ($|\hat{Y}_t - \hat{Y}_{t-1}| < 0.015$).
4. **Layer 4 — Batch Experimentation & RQ1–RQ7 Statistical Suite (`pipelines/`):**
   - Ablation-pure evaluation across `cot_persona`, `few_shot_persona`, `feature_informed` (raw 1-turn LLM baselines), and `agentic` (tool-assisted closed loop).
   - Automated multi-granularity proximity analysis (`pipelines/analyze_experimentation.py`) and complete RQ1–RQ7 statistical & qualitative evaluation (`pipelines/run_rq_statistical_analysis.py`).

---

## Installation & Vocalization ONNX Setup (Required Before Running)

### 1. Install Package & Dependencies
```bash
git clone [https://github.com/g0ldata/heb-text-lvl.git](https://github.com/g0ldata/heb-text-lvl.git)
cd heb-text-lvl
pip install -r requirements.txt
pip install -e .
```
### 2. Download the Quantized Vocalization ONNX Model (dicta-1.0.int8.onnx)
Both the feature extractor (which vocalizes Hebrew text to count syllables) and the agentic vocalizer require the 293.2 MB dicta-1.0.int8.onnx model inside .cache/dicta-1.0.int8.onnx.
Before running inference or experimentation, download the verified binary directly via wget:
```bash
mkdir -p .cache
wget -O .cache/dicta-1.0.int8.onnx https://github.com/thewh1teagle/dicta-onnx/releases/download/model-files-v1.0/dicta-1.0.int8.onnx
ls -lh .cache/dicta-1.0.int8.onnx
```
(Verify that .cache/dicta-1.0.int8.onnx is ~294 MB / 307,492,637 bytes. ensure_dicta_onnx_model() in src/hebrew_readability/hf_hub.py also enforces a >=250 MB integrity gate automatically.)

### 3. Configure Environment Variables
Copy .env.example to .env (or set Colab Secrets) with your Hugging Face and OpenAI API keys:
```bash
cp .env.example .env
# Set HF_TOKEN and OPENAI_API_KEY
```
## Running Experimentation & Statistical Analysis
```bash
# 1. Run the 4-method experimentation pipeline (gpt-4o)
python pipelines/run_experimentation.py \
    --input-csv artifacts/wikipedia_simplification_test_100.csv \
    --output-csv artifacts/experimentation_results.csv \
    --model gpt-4o \
    --max-iters 3

# 2. Run proximity analysis, 14x4 per-level tables & paired McNemar/Wilcoxon tests
python pipelines/analyze_experimentation.py \
    --results-csv artifacts/experimentation_results_enriched.csv \
    --output-dir artifacts/analysis

# 3. Run the complete RQ1-RQ7 statistical & qualitative evaluation suite
python pipelines/run_rq_statistical_analysis.py \
    --results-csv artifacts/experimentation_results_enriched.csv \
    --trajectories-jsonl artifacts/agent_trajectories.jsonl \
    --output-dir artifacts/analysis
```
---
