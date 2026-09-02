Document Forgery Detection — CNN Classifier with Grad-CAM & ELA Interpretability
A transfer-learning image forensics pipeline that detects tampered images, benchmarks two CNN backbones, and empirically demonstrates why models trained on generic photo tampering fail to transfer to real document forgery — with interpretability via Grad-CAM and classical Error Level Analysis (ELA).
Motivation
Most public image-tampering-detection projects train and evaluate exclusively on CASIA v2.0, a dataset of natural-photo splicing and copy-move forgeries (a bird pasted into a different sky, a cloned tree, etc.). This is a well-trodden recipe — dozens of near-identical repos exist. Document fraud (edited invoices, altered receipts, tampered IDs) is a visually and structurally different problem: forged regions are small, localized edits on flat, text-heavy backgrounds, not large-scale scene manipulation.
This project's core question: does a model trained on generic image tampering actually transfer to real document forgery, or is that assumption wrong? The answer, measured empirically below, is a clear no — and that negative result, backed by numbers, is the project's main contribution.
Project Structure
1. CASIA baseline           → Compare EfficientNet-B0 vs ResNet50 (transfer learning, frozen backbone)
2. Fine-tuning               → Unfreeze ResNet50's last block, recover additional accuracy
3. Domain-shift evaluation   → Test the CASIA-trained model on REAL document forgery, zero-shot
4. Document fine-tuning      → Attempt to recover performance on document data; diagnose why it doesn't fully work
5. Interpretability          → Grad-CAM (CNN attention) + ELA (classical forensics) side by side
1. Baseline: EfficientNet-B0 vs ResNet50 on CASIA v2.0
Dataset: CASIA v2.0 — 12,614 images (7,491 authentic, 5,123 tampered; copy-move and splicing forgeries).
Setup: ImageNet-pretrained backbones, frozen feature extractor, only the classification head trained. 80/10/10 stratified train/val/test split. 15 epochs, Adam, lr=1e-4.
Metric	EfficientNet-B0	ResNet50
Test Accuracy	67.6%	71.6%
Test F1	0.553	0.617
Test AUC	0.729	0.794
Finding: ResNet50 outperformed EfficientNet-B0 on every metric, despite EfficientNet-B0 typically matching or beating ResNet50 on standard ImageNet benchmarks — a specific, non-obvious result for this task. ResNet50 was carried forward as the primary model.
2. Fine-tuning ResNet50 (unfreezing layer4)
The frozen-backbone baseline only trained the final classification head, so the pretrained ImageNet features never adapted to tampering-specific patterns. layer4 (the last residual block) was unfrozen and fine-tuned at a low learning rate (1e-5) for 10 additional epochs.
Metric	Frozen baseline	Fine-tuned (layer4 unfrozen)	Change
Accuracy	73.9%	76.5%	+2.6 pts
F1	0.658	0.720	+0.062
AUC	0.829	0.847	+0.019
Finding: Letting the last convolutional block adapt to tampering-specific patterns produced a real, consistent improvement across every metric — F1 in particular improved meaningfully, indicating better balance between catching tampered images and avoiding false alarms.
3. Domain-Shift Evaluation: Zero-Shot on Real Document Forgery
Dataset: "Find it again!" — the ICDAR 2023 receipt forgery dataset, built from real scanned SROIE receipts (987 usable images, ~16.5% tampered: edited prices, dates, item descriptions).
The fine-tuned CASIA model was evaluated on this dataset with zero retraining — the same weights, pointed at a completely different kind of tampering.
Metric	CASIA (source domain)	Documents (target domain, zero-shot)	Gap
Accuracy	76.5%	81.0%¹	+4.5 pts
Precision	0.695	0.067	−0.629
Recall	0.746	0.012	−0.734
F1	0.720	0.021	−0.699
AUC	0.847	0.527	−0.320
¹ Accuracy is misleading here — see below.
Finding — the headline result of this project: accuracy alone suggests the model did fine. It didn't. The confusion matrix reveals the model predicted "authentic" for nearly every image, catching only 2 out of 162 actually-tampered documents:
                predicted authentic   predicted tampered
true authentic         797                    28
true tampered           160                     2
An AUC of 0.527 is statistically indistinguishable from random guessing (AUC = 0.5). The model didn't degrade gracefully — it lost virtually all discriminative power for this task. High accuracy on an imbalanced dataset (~84% authentic) is a classic trap: a model that always predicts the majority class scores well on accuracy while being functionally useless. This independently reproduces a pattern described in recent document-forensics research: forged regions in documents occupy a tiny fraction of pixels (well under 5%, an order of magnitude smaller than natural-image tampering), and models built around large-region evidence may carry no signal for this at all once images are resized to typical CNN input resolutions (e.g. 224×224).
4. Attempted Recovery: Fine-tuning on Document Data
The model was fine-tuned on the "Find it again!" dataset's own train split (~690 images, ~110 tampered), using the same layer4-unfreezing approach as before — this time with a class-weighted loss to counter the ~16.5% tampered class imbalance (otherwise the model would trivially re-learn "always predict authentic" and still show a deceptively fine accuracy).
Metric	Before fine-tuning (zero-shot)	After fine-tuning	Change
Accuracy	81.2%	58.7%	−22.5 pts
Precision	0.000	0.169	+0.169
Recall	0.000	0.400	+0.400
F1	0.000	0.237	+0.237
AUC	0.592	0.492	−0.100
Finding: Fine-tuning forced a real behavior change — recall went from 0% to 40%, meaning the model started actually flagging tampered documents instead of ignoring the class entirely. But precision stayed low (69 false positives vs. 14 true positives) and AUC remained at/below chance level throughout training (oscillating between 0.44–0.49 across all 15 epochs) — indicating the model never found a genuinely reliable discriminative signal. It shifted from "confidently wrong in one direction" to "noisily wrong in another," rather than actually learning to detect document tampering.
Why this is the expected outcome, not a bug: with only ~110 tampered training examples, and — per the finding in Section 3 — tampering evidence that may already be destroyed by resizing images down to 224×224 before the model even sees them, there may simply be no recoverable signal left in the input the model receives. This suggests that reliably detecting document forgery likely requires either much more training data, higher input resolution, or a patch-based / localization approach that examines document regions at native resolution rather than a single downsampled global classification — which is where current document-forensics research is actually focused, rather than standard whole-image CNN classification.
5. Interpretability: Grad-CAM + Error Level Analysis (ELA)
For 8 representative CASIA test images (2 each of true positives, true negatives, false negatives, and false positives), two complementary visualizations were generated side by side:
Grad-CAM — gradient-weighted class activation mapping on layer4, showing which regions the CNN's prediction was most sensitive to.
ELA (Error Level Analysis) — a classical, non-deep-learning forensic technique: the image is re-compressed at a fixed JPEG quality and diffed against the original. Edited regions often show a different compression error level than untouched pixels, since they underwent an extra round of compression the rest of the image didn't.
Including both false positives and false negatives (not just successful predictions) alongside the classical ELA baseline was a deliberate choice — it shows where and why the model fails, and how its learned attention compares to an established, interpretable forensic heuristic, rather than only showcasing favorable cases.
Observation: Grad-CAM heatmaps were meaningfully varied across images — attention consistently localized to specific objects, faces, or structural regions rather than defaulting to a generic center-of-image pattern, suggesting the model learned genuine, image-specific features rather than a trivial shortcut. ELA maps, by contrast, were mostly faint/noisy without clearly localized signal on this dataset. This is a known limitation of ELA outside controlled conditions: it relies on a known, consistent JPEG compression history, and CASIA v2.0 mixes JPEG and TIFF sources with variable prior compression and resizing — which degrades the technique's reliability. This is itself a useful finding: classical single-technique forensics (ELA) and learned CNN attention (Grad-CAM) do not always agree, and combining them into an ensemble (rather than treating either as sufficient alone) is a natural next step.
(See gradcam_ela_contact_sheet.png for the full visualization grid.)
Key Takeaways
ResNet50 outperformed EfficientNet-B0 for this specific tampering-classification task, despite the opposite being common on general benchmarks — backbone choice matters and shouldn't be assumed.
Fine-tuning (unfreezing the last block) provides a real, measurable improvement over a frozen-backbone baseline, even on a modest ~12k-image dataset.
Generic image-tampering detectors do not transfer to document forgery. A model with strong CASIA performance (F1 0.72) collapsed to near-random discriminative power (AUC 0.53, F1 0.02) on real document data — this is the project's central, empirically-grounded finding.
Naive fine-tuning on limited document data does not fix the underlying problem. It can shift model behavior (e.g. increase recall) without producing a genuinely reliable detector, most likely because standard image resizing destroys the small-scale evidence document tampering depends on.
Document forgery detection likely needs a fundamentally different approach — higher resolution, patch-based analysis, or localization rather than whole-image classification — consistent with where current research in this space is heading.
Tech Stack
PyTorch / torchvision — model training, transfer learning (ResNet50, EfficientNet-B0)
pytorch-grad-cam — Grad-CAM implementation
scikit-learn — metrics, stratified train/val/test splitting
Pillow — image I/O, ELA implementation
Kaggle GPU (P100) — training compute
Datasets
CASIA v2.0 Image Tampering Detection Dataset (Kaggle, by divg07)
Find it again! Dataset (Kaggle, by nikita2998) — ICDAR 2023 receipt forgery, built from SROIE
Possible Future Work
Patch-based / localization approach (predict tampered regions, not just image-level labels) using the pixel-level ground-truth masks available in both CASIA2 and the "Find it again!" dataset
Higher-resolution or multi-scale input pipeline to preserve small-scale document tampering evidence
Larger document-forgery dataset (e.g. DocTamper, 120k images) for a more data-sufficient fine-tuning attempt
Combine ELA and CNN predictions into an ensemble rather than treating them as separate visualizations
