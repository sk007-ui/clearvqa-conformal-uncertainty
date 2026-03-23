# Paper Reproduction

This folder contains results from reproducing the paper:
"Benchmarking LLMs via Uncertainty Quantification" (arxiv 2401.12794)

## Purpose
Before applying conformal uncertainty to ClearVQA, we verified 
the original codebase by reproducing the paper's results.

## Models Tested
- Llama-2-7B
- Mistral-7B

## Result
All accuracy and set size numbers match the paper exactly.
This confirms the codebase is correct and reliable.

## How to Reproduce
```
python uncertainty_quantification_via_cp.py \
  --model Llama-2-7b-hf \
  --raw_data_dir data \
  --logits_data_dir outputs_base \
  --cal_ratio 0.5 \
  --alpha 0.1
```
