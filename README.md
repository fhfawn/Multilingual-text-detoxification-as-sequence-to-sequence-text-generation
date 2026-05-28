# Multilingual Text Detoxification with Transformer Seq2Seq Models

This repository contains a Deep Learning project for **multilingual text detoxification**.  
The goal is to rewrite toxic sentences into non-toxic, fluent sentences while preserving the original meaning and language.

Team: Iman Chantieva, Abdul-Malik Khatuev, Irina Brodskaya, Akmuhammet Gurbangeldiyev

---

## 1. Project Goal

Text detoxification is a controlled text generation task:

```text
Input:  toxic sentence in language L
Output: non-toxic sentence in the same language L, preserving meaning
```

In this project, detoxification is formulated as a **sequence-to-sequence generation problem** using Transformer-based models. The system compares simple baselines, pretrained detoxification models, and a parameter-efficient fine-tuned model.

The main questions are:

1. Can a multilingual Transformer model learn to rewrite toxic sentences into neutral paraphrases?
2. How much does fine-tuning improve over rule-based and pretrained baselines?
3. Can candidate generation and reranking improve the trade-off between toxicity reduction and meaning preservation?
4. How stable is the system across different languages?

---

## 2. Dataset

The main dataset used in the notebook is from the TextDetox / Multilingual ParaDetox family:

- Hugging Face dataset: <https://huggingface.co/datasets/textdetox/multilingual_paradetox>
- Test dataset variant: <https://huggingface.co/datasets/textdetox/multilingual_paradetox_test>
- TextDetox organization: <https://huggingface.co/textdetox>
- PAN 2024 Multilingual Text Detoxification task: <https://pan.webis.de/clef24/pan24-web/text-detoxification.html>

The dataset contains parallel examples of toxic and neutral sentences. Each training example is expected to contain:

| Field | Meaning |
|---|---|
| `toxic_sentence` / source text column | Toxic input sentence |
| `neutral_sentence` / target text column | Human-written or curated non-toxic rewrite |
| `lang` / language column | Language code |

In the current run, the notebook loaded:

```text
3600 examples
9 languages
2880 train examples
720 validation examples
```

The notebook is written defensively and tries several dataset candidates:

```python
DATASET_CANDIDATES = [
    "textdetox/multilingual_paradetox",
    "textdetox/multilingual_paradetox_train",
    "textdetox/multilingual_paradetox_test",
]
```

If a local dataset is used, it can be provided through:

```python
LOCAL_DATA_PATH = "/path/to/train.csv"
```

---

## 3. Related Research and Background

### Text detoxification

The project is based on the idea of detoxification with parallel data, where toxic sentences are paired with non-toxic paraphrases.

Key references:

1. **ParaDetox: Detoxification with Parallel Data**  
   ACL 2022 paper: <https://aclanthology.org/2022.acl-long.469/>  
   GitHub repository: <https://github.com/s-nlp/paradetox>

2. **MultiParaDetox: Extending Text Detoxification with Parallel Data to New Languages**  
   arXiv: <https://arxiv.org/abs/2404.02037>

3. **SmurfCat at PAN 2024 TextDetox: Alignment of Multilingual Transformers for Text Detoxification**  
   Hugging Face paper page: <https://huggingface.co/papers/2407.05449>  
   GitHub repository: <https://github.com/s-nlp/multilingual-transformer-detoxification>

4. **TextDetox Shared Task / Multilingual ParaDetox**  
   Dataset page: <https://huggingface.co/datasets/textdetox/multilingual_paradetox>  
   PAN 2024 task page: <https://pan.webis.de/clef24/pan24-web/text-detoxification.html>

### Transformer and seq2seq modeling

The project uses Transformer-based text-to-text generation. The main architecture family is based on the encoder-decoder Transformer, which is suitable for tasks such as translation, summarization, paraphrasing, and detoxification.

Key references:

1. **Attention Is All You Need**  
   Vaswani et al., 2017: <https://arxiv.org/abs/1706.03762>

2. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)**  
   Raffel et al., 2020: <https://arxiv.org/abs/1910.10683>

3. **mT5: A massively multilingual pre-trained text-to-text transformer**  
   Xue et al., 2020: <https://arxiv.org/abs/2010.11934>

### Parameter-efficient fine-tuning

The project uses LoRA as the main trainable adaptation method. LoRA freezes the base model and trains a small set of low-rank adapter weights, reducing the number of trainable parameters and GPU memory requirements.

Key references:

1. **LoRA: Low-Rank Adaptation of Large Language Models**  
   Hu et al., 2021: <https://arxiv.org/abs/2106.09685>

2. **PEFT library by Hugging Face**  
   Documentation: <https://huggingface.co/docs/peft/index>

### Evaluation of generated text

The project evaluates generation quality along multiple dimensions:

- toxicity reduction;
- semantic preservation;
- fluency proxy;
- reference similarity;
- per-language robustness.

Useful references:

1. **BERTScore: Evaluating Text Generation with BERT**  
   Zhang et al., 2019: <https://arxiv.org/abs/1904.09675>

2. **BLEU: a Method for Automatic Evaluation of Machine Translation**  
   Papineni et al., 2002: <https://aclanthology.org/P02-1040/>

---

## 4. Project Pipeline

The notebook is organized as a full research pipeline.

### Step 1. Environment setup

The notebook installs and imports the required libraries:

- `transformers`
- `datasets`
- `evaluate`
- `sentence-transformers`
- `peft`
- `accelerate`
- `bitsandbytes`, if available
- `pandas`, `numpy`, `torch`, `tqdm`

It also includes memory-safe GPU configuration:

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

and a utility function:

```python
clear_cuda_memory()
```

This is important because several models may be loaded during the same notebook session.

---

### Step 2. Data loading and preprocessing

The notebook loads the multilingual detoxification dataset, detects source, reference, and language columns, and normalizes them into a common format:

```text
toxic_input
neutral_reference
lang
```

Then it performs a train/validation split:

```text
80% train
20% validation
```

The split is stratified by language when possible.

---

### Step 3. Exploratory data analysis

The notebook reports:

- number of examples;
- number of languages;
- examples per language;
- average input length;
- availability of neutral references.

This is used to understand whether the dataset is balanced and whether the model should be evaluated per language.

---

## 5. Models and Methods

### 5.1 Copy baseline

The simplest baseline returns the input unchanged:

```python
copy_baseline = toxic_input
```

This baseline should preserve meaning perfectly but should not reduce toxicity.

Purpose:

- gives a lower bound for detoxification;
- helps check whether toxicity metrics are meaningful.

---

### 5.2 Rule-based detoxification baseline

The rule-based baseline uses toxic lexicons and handcrafted replacements. It removes or replaces toxic words in the input sentence.

Example logic:

```python
toxic word -> neutral replacement
toxic word without replacement -> removed
```

Purpose:

- simple and interpretable baseline;
- useful for comparison with neural models;
- often reduces explicit toxicity but can damage grammar and meaning.

Limitations:

- weak for implicit toxicity;
- weak for morphology-rich languages;
- weak for languages with limited lexicon coverage;
- may over-delete important content.

---

### 5.3 Pretrained seq2seq detoxification baseline

The notebook tries several pretrained or fallback seq2seq models:

```python
PRETRAINED_DETOX_MODEL_CANDIDATES = [
    "s-nlp/bart-base-detox",
    "s-nlp/ruT5-base-detox",
    "google/mt5-small",
    "s-nlp/mt0-xl-detox-orpo",
]
```

The large multilingual detoxification model is:

- <https://huggingface.co/s-nlp/mt0-xl-detox-orpo>

However, this model has around 3.7B parameters and can be too large for 14-16 GB GPUs. The notebook therefore uses memory-safe loading and falls back to smaller models when necessary.

The function responsible for this is:

```python
load_pretrained_detox_model()
```

Purpose:

- compare project models against existing pretrained detoxification models;
- keep the pipeline runnable on limited hardware.

---

### 5.4 Main model: mT5-small + LoRA fine-tuning

The main trainable model is:

```python
BASE_SEQ2SEQ_MODEL = "google/mt5-small"
```

It is fine-tuned on toxic-neutral sentence pairs using the following prompt format:

```text
detoxify {lang}: {toxic_sentence}
```

The target output is:

```text
neutral_sentence
```

The notebook uses LoRA by default:

```python
USE_LORA = True
```

LoRA configuration:

```python
LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q", "v"],
)
```

In the observed run, the model reported approximately:

```text
trainable params: 344,064
all params: 556,635,520
trainable%: 0.0618
```

This means only a very small fraction of parameters is trained, which makes the experiment suitable for limited GPU resources.

---

## 6. Candidate Generation and Reranking

The project also implements a multi-objective reranking stage.

Instead of using only one generated output, the model can generate several candidates:

```text
candidate_1
candidate_2
candidate_3
...
```

Each candidate is scored by a combination of:

1. toxicity score;
2. semantic similarity to the original input;
3. length penalty;
4. repetition penalty or fluency proxy.

The final reranking objective is conceptually:

```text
score = alpha * semantic_similarity
      - beta  * toxicity_score
      - gamma * length_penalty
      - delta * repetition_penalty
```

The best candidate is selected as the final detoxified output.

Purpose:

- reduce toxicity while preserving meaning;
- avoid outputs that are too short;
- avoid outputs that simply delete toxic content;
- make decoding more controllable.

---

## 7. Evaluation

The notebook does not rely on leaderboard submission. Instead, it performs automatic evaluation and analysis.

### 7.1 Automatic metrics

The evaluation includes:

| Metric | Direction | Meaning |
|---|---:|---|
| Toxicity score | lower is better | How toxic the output remains |
| Toxicity reduction | higher is better | Difference between input and output toxicity |
| Semantic similarity | higher is better | Whether the original meaning is preserved |
| Reference similarity | higher is better | Similarity to human neutral rewrite |
| Length ratio | close to 1 is better | Whether output is not over-deleted |
| Repetition / fluency proxy | lower is better | Whether generation is repetitive or unnatural |

### 7.2 Per-language evaluation

The notebook groups evaluation results by language:

```python
per_language_evaluation.csv
```

This is important because multilingual models can perform very differently across high-resource and low-resource languages.

### 7.3 Error analysis

The notebook automatically samples examples for qualitative inspection and assigns potential error types such as:

- toxicity not removed;
- meaning changed;
- output too short;
- copied input;
- hallucinated content;
- wrong language;
- broken fluency.

This section supports the final discussion and helps explain metric results.

---

## 8. Output Files

The notebook writes analysis artifacts to:

```text
outputs_text_detox_project/
```

Expected files:

```text
outputs_text_detox_project/
├── automatic_evaluation_results.csv
├── model_outputs_for_analysis.csv
├── per_language_evaluation.csv
└── mt5_detox_lora_final/
```

Where:

| File | Description |
|---|---|
| `automatic_evaluation_results.csv` | Row-level metric results for each model |
| `model_outputs_for_analysis.csv` | Inputs, references, and model outputs |
| `per_language_evaluation.csv` | Aggregated metrics by language |
| `mt5_detox_lora_final/` | Saved LoRA/fine-tuned model checkpoint |

---

## 9. How to Run

### Option 1. Run only baselines and evaluation

Use this configuration:

```python
RUN_PRETRAINED_INFERENCE = True
RUN_TRAINING = False
RUN_RERANKING = True
```

This is faster and safer for limited GPUs.

### Option 2. Run LoRA fine-tuning

Use this configuration:

```python
RUN_PRETRAINED_INFERENCE = True
RUN_TRAINING = True
RUN_RERANKING = True
```

Recommended hardware:

```text
GPU with at least 14-16 GB VRAM
```

The notebook uses memory-safe settings:

```python
BATCH_SIZE = 1
GENERATION_NUM_BEAMS = 1
RERANK_NUM_BEAMS = 3
RERANK_NUM_RETURN_SEQUENCES = 3
```

Fine-tuning uses small per-device batches with gradient accumulation.

---

## 10. Reproducibility Notes

The notebook sets a random seed and uses deterministic splits where possible.

Important configuration variables:

```python
BASE_SEQ2SEQ_MODEL = "google/mt5-small"
SENTENCE_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOXICITY_CLASSIFIER_MODEL = "unitary/multilingual-toxic-xlm-roberta"

MAX_INPUT_LENGTH = 96
MAX_TARGET_LENGTH = 96

BATCH_SIZE = 1
RUN_TRAINING = True
```

If GPU memory is limited, set:

```python
RUN_TRAINING = False
```

or reduce the number of evaluation examples.

---

## 11. Results Interpretation

The expected behavior is:

1. **Copy baseline**  
   Preserves meaning but keeps toxicity.

2. **Rule-based baseline**  
   Reduces explicit toxic words but can break grammar and remove important meaning.

3. **Pretrained detoxification baseline**  
   Produces more fluent generations but may be language-dependent.

4. **Fine-tuned mT5 + LoRA**  
   Learns the project-specific toxic-to-neutral mapping while training only a small number of parameters.

5. **Reranking**  
   Improves the balance between toxicity reduction and semantic preservation by choosing the best candidate among several generations.

---

## 12. Limitations

The current system has several limitations:

1. Automatic toxicity classifiers may be biased or unreliable across languages.
2. Semantic similarity metrics may not detect subtle meaning shifts.
3. Rule-based detoxification is brittle for morphology-rich and low-resource languages.
4. Some pretrained detoxification models are too large for limited GPUs.
5. The model may over-sanitize text by deleting emotional or stylistic content.
6. Human evaluation is not included, although it would be the best way to assess generation quality.

---

## 13. Future Work

Possible improvements:

1. Add human evaluation for toxicity, meaning preservation, and fluency.
2. Tune reranking weights on a validation set.
3. Compare mT5-small with mBART, mT0, and Aya models.
4. Add backtranslation-based data augmentation.
5. Add language-specific prompts and compare them with a unified prompt format.
6. Use 4-bit QLoRA for larger models.
7. Add calibration analysis for toxicity classifiers.
8. Evaluate robustness on unseen languages.

---

## 14. Short Project Summary

This project implements a multilingual text detoxification pipeline using Transformer-based sequence-to-sequence models. It compares copy, rule-based, pretrained, and LoRA fine-tuned models. The project evaluates outputs using toxicity reduction, semantic preservation, reference similarity, fluency proxies, and per-language analysis. The final system is designed as a Deep Learning research project rather than a leaderboard submission pipeline.

