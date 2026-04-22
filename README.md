# Benchmark: Conformal Uncertainty in Vision-Language Models (VLMs)

This repository contains the official evaluation pipeline for quantifying uncertainty in Vision-Language Models using **Conformal Prediction (CP)**. Conducted as part of a research internship at the **Indian Institute of Science (IISc), Bangalore**.

## 🚀 Project Overview
Vision-Language Models (like Qwen2-VL and LLaVA) are often overconfident in their predictions. This project implements a mathematical wrapper using **LAC (Least Ambiguous Class)** and **APS (Adaptive Prediction Sets)** to provide a 90% statistical guarantee that the true answer is contained within the prediction set.

## 📊 Evaluation Axis
We evaluate two different architectures across three distinct types of visual ambiguity:
1. **Perceptual Ambiguity (ClearVQA):** Basic visual noise and blur.
2. **Situational Ambiguity (VCR):** Complex social reasoning and intent.
3. **Knowledge Ambiguity (A-OKVQA):** Retrieval of outside-world knowledge.

## 🤖 Models Tested
- **Qwen2-VL-2B** (Baseline)
- **LLaVA-1.5-7B** (Heavyweight / 4-bit NF4)

## 📈 Key Findings
- **Calibration:** The CP framework successfully maintained **~90% coverage** ($\alpha=0.1$) across all datasets.
- **Set Expansion:** As visual blur increases (Radius 0 to 8), prediction sets expand significantly to maintain reliability.
- **Knowledge Collapse:** A-OKVQA showed the highest uncertainty spike (52% for LLaVA), proving that cognitive retrieval relies heavily on visual clarity.

## 📁 Repository Structure
- `run_*.py`: Inference scripts for different datasets and models.
- `eval_*.py`: Conformal Prediction evaluation logic (Calibration & Testing).
- `plot_*.py`: Publication-quality visualization scripts.
- `cp_results.json`: Detailed metrics for all experiments.

---
*Internship Supervisor: Sambit (PhD Candidate, IISc)*
