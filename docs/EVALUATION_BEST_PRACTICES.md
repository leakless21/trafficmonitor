# Best Practices for Model Evaluation and Comparison

### Focus: Vehicle Counting and License-Plate Recognition Pipelines

> **Purpose** – Provide a concise, actionable checklist to ensure that experiments are fair, reproducible and informative when benchmarking detection + OCR systems in traffic-monitor projects.

---

## 1 Dataset Preparation

1. **Curate diverse data**  
   • Cover day/night, weather, camera angles, motion blur and occlusion scenarios.  
   • Include edge cases (dirty plates, motorcycles, tail lights, etc.).
2. **Annotation quality**  
   • Double-annotate ≥10 % of the dataset and resolve disagreements.  
   • Use polygon masks for plates if you plan to test segmentation methods.
3. **Train/Val/Test split**  
   • Split by _video_ or _camera_ rather than by frame to avoid temporal leakage.  
   • Keep the test set _frozen_ once baselines are published.
4. **Stratified sampling**  
   • Maintain class ratios (vehicle classes, country plate styles).
5. **Version datasets**  
   • Tag releases (e.g. `v1.2`) and log SHA hashes in reports.

---

## 2 Evaluation Metrics

### 2.1 Object & Plate Detection

| Category        | Metric                     | Rationale                                                                    |
| --------------- | -------------------------- | ---------------------------------------------------------------------------- |
| Accuracy        | mAP50-95                   | Area-under-curve view of precision-recall across IoU thresholds (COCO style) |
| Error Structure | Confusion matrix, PR curve | Reveals systematic misclassifications                                        |
| Speed           | Latency (ms/im), FPS       | Deployment constraint                                                        |
| Efficiency      | Model size (MB), FLOPs     | Edge-device feasibility                                                      |

### 2.2 Vehicle Counting

| Metric                       | Formula                                   | Notes                    |
| ---------------------------- | ----------------------------------------- | ------------------------ | -------- | ------------------------------ |
| Mean Absolute Error (MAE)    | \(\frac{1}{N}\sum                         | \hat c_i - c_i           | \)       | Robust to over/under counts    |
| Mean Percentage Error (MAPE) | \(\frac{1}{N}\sum \frac{                  | \hat c_i-c_i             | }{c_i}\) | Scales error to traffic volume |
| Recall@Window                | % of objects counted in a temporal window | Measures delay tolerance |

### 2.3 License-Plate OCR

| Category                  | Metric                                             | Rationale                             |
| ------------------------- | -------------------------------------------------- | ------------------------------------- |
| Character Error Rate      | Levenshtein distance / #chars                      | Fine-grained text quality             |
| Plate-level Accuracy & F1 | Exact string match; handles insertions + deletions |
| Detection Rate            | Plates detected / total                            | Separates localisation vs recognition |
| Throughput                | Plates ✓ / sec                                     | Critical for real-time tolling        |

---

## 3 Experimental Protocol

1. **Fixed seeds** – Set `torch.manual_seed`, `numpy.random.seed`, `random.seed`.
2. **Deterministic inference** – Disable drop-path, dropout, cudnn nondeterminism.
3. **Warm-up** – Discard first N batches when timing (GPU clocks up).
4. **Batch size = 1 for latency**, large batch for throughput.
5. **Single variable principle** – Change _one_ factor per ablation (model, input size, quantisation).
6. **Repeat runs** – ≥3 runs; report _mean±std_.
7. **Hardware & software lock** – Document GPU model, driver, CUDA; pin library versions in `requirements.txt`.
8. **Config as code** – Store YAML/JSON configs under version control.

---

## 4 Result Reporting Guidelines

1. **Table + plot** – Present both a summary table and visual graphs (bar, PR, CM).
2. **Highlight trade-offs** – Use colour coding or radar charts to emphasise speed vs accuracy.
3. **Qualitative samples** – Include success & failure frames; annotate why failures occur (glare, motion blur…).
4. **Statistical significance** – Use paired t-test or bootstrap when comparing small deltas (<1 pp mAP).
5. **Provide raw logs** – Attach `*.log` and JSON metric dumps for transparency.

---

## 5 Reproducibility Checklist (adapted from ML Reproducibility Challenge)

- [x] Code submitted or linked in public repo
- [x] Dataset access instructions
- [x] Exact commit/weights used
- [x] Hardware spec (CPU/GPU/TPU)
- [x] Time & cost to train/infer
- [x] Random seed values
- [x] Licenses for code + data

---

## 6 Deployment & Format Comparison

1. **Export variants** – TorchScript, ONNX, TensorRT, CoreML, NCNN, MNN.
2. **Accuracy drift test** – Run benchmark suite after each conversion; flag if >1 pp drop.
3. **Quantisation** – Calibrate with ≥500 samples; validate accuracy; measure int8 speed-up.
4. **Edge devices** – Evaluate power (W) & memory (RAM) footprints.

---

## 7 Ethical & Legal Considerations

1. **Privacy** – Blur faces, follow GDPR/local regulations.
2. **Bias** – Check plate OCR accuracy across regions (e.g. EU vs US plates).
3. **License compliance** – Observe upstream model/data licenses (e.g. Ultralytics YOLO AGPL).
4. **Responsible deployment** – Prevent misuse (e.g. surveillance beyond vehicle counting).

---

## 8 Suggested Further Reading

1. Lin et al., _Microsoft COCO: Common Objects in Context_, ECCV 2014.
2. Redmon & Farhadi, _YOLOv3: An Incremental Improvement_, 2018.
3. Zhang et al., _Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline_, ECCV 2020.
4. NIST, _Face Recognition Vendor Test_ – methodology inspiration for OCR benchmarking.
5. Papers With Code _Model Efficiency Toolkit_ – practical tips on benchmarking hardware.

---

### How to Use This Document

Copy this file into your thesis appendix or methodology chapter. Use the checkboxes as you design each experiment to avoid common pitfalls and improve credibility.
