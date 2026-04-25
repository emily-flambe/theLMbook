# Marimo Notebooks

Notebooks have been converted from Jupyter (`.ipynb`) to [marimo](https://marimo.io) (`.py`) and reorganized by book chapter.

## Chapter 2 — Tokenization & Count-Based Models

- [2.3. Byte-Pair Encoding](https://github.com/aburkov/theLMbook/blob/main/chapter-02/byte_pair_encoding.py)
- [2.5. Count-Based Language Model](https://github.com/aburkov/theLMbook/blob/main/chapter-02/count_language_model.py)

## Chapter 3 — Recurrent Networks

- [3.6. Training an RNN Language Model](https://github.com/aburkov/theLMbook/blob/main/chapter-03/news_RNN_language_model.py)

## Chapter 4 — Transformers

- [4.9. Transformer in Python](https://github.com/aburkov/theLMbook/blob/main/chapter-04/news_decoder_language_model.py)

## Chapter 5 — Finetuning, Sampling & Classification

- [5.3.1. Baseline Emotion Classifier](https://github.com/aburkov/theLMbook/blob/main/chapter-05/emotion_classifier_LR.py)
- [5.3.2. Emotion Generation](https://github.com/aburkov/theLMbook/blob/main/chapter-05/emotion_GPT2_as_text_generator.py)
- [5.3.3. Finetuning to Follow Instructions](https://github.com/aburkov/theLMbook/blob/main/chapter-05/instruct_GPT2.py)
- [5.4.4. Penalties](https://github.com/aburkov/theLMbook/blob/main/chapter-05/sampling_method.py)
- [5.5.2. Parameter-Efficient Finetuning (PEFT)](https://github.com/aburkov/theLMbook/blob/main/chapter-05/emotion_GPT2_as_text_generator_LoRA.py)
- [5.6. LLM as a Classifier](https://github.com/aburkov/theLMbook/blob/main/chapter-05/emotion_GPT2_as_classifier.py)

## Extras (supplementary tutorials)

Notebooks not tied to a specific book section:

- [Emotion Classifier (CNN baseline)](https://github.com/aburkov/theLMbook/blob/main/extras/emotion_classifier_CNN.py)
- [Document Classifier with LLMs as Labelers](https://github.com/aburkov/theLMbook/blob/main/extras/document_classifier_with_LLMs_as_labelers.py)
- [GRPO with Qwen-0.5B-Instruct](https://github.com/aburkov/theLMbook/blob/main/extras/GRPO_Qwen_0_5_Instruct.py)
- [GRPO from Scratch (Multi-GPU, Qwen-2.5-1.5B-Instruct)](https://github.com/aburkov/theLMbook/blob/main/extras/GRPO_From_Scratch_Multi_GPU_DataParallel_Qwen_2_5_1_5B_Instruct.py)

## Running a notebook

```bash
make install          # creates .venv, installs marimo + ML deps
make marimo           # opens marimo in the browser
# or directly:
.venv/bin/marimo edit chapter-02/byte_pair_encoding.py
```
