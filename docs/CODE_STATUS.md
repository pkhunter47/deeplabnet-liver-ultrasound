# Code status

The original training implementation was not present in the uploaded research package. The code in this repository is a clean-room, manuscript-derived reference that captures the disclosed design: a DeepLabv3–MobileNetV3 backbone, fixed directional Haar features, predictive-entropy conditioning, residual mask refinement, and boundary/distance auxiliary heads.

It does **not** establish reproduction of the reported Dice, IoU, parameter count, or robustness results. Reproduction requires the original 732-image dataset, the duplicate-group-disjoint split manifest, exact polygon rasterization, augmentation settings, loss weights, random seeds, and checkpoint-selection rule. Replace this implementation with the recovered original only after checking those items and removing private data or credentials.
