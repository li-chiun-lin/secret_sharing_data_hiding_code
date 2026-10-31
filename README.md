# Lossless Meaningful Secret Image Sharing via Shuffled Mixed-Depth Modified Least-Significant-Bit Substitution

Official C++ research testbed, benchmark engine, and steganalysis suite for the research paper:

> **Lossless Meaningful Secret Image Sharing via Shuffled Mixed-Depth Modified Least-Significant-Bit Substitution**  
> *Li-Chiun Lin, Cheng-Hsing Yang, Cheng-Ta Huang, and Shiuh-Jeng Wang*  
> Submitted to *Journal of Information Security and Applications* (Elsevier), under review, 2026.

---

## Overview

In meaningful Secret Image Sharing (SIS), noise-like shares are concealed inside recognizable cover images to mitigate eavesdropping risks. However, standard uniform-depth embedding typically degrades cover image fidelity and leaves detectable boundary artifacts across spatial blocks.

This repository provides an end-to-end C++ implementation of a meaningful SIS scheme that mathematically guarantees lossless reconstruction of the secret image. The design integrates:
1. Finite-field polynomial sharing over Galois Field $GF(2^8)$ with constant-polynomial preprocessing.
2. A shuffled mixed-depth Modified Least-Significant-Bit (MLSB) allocation strategy.
3. The Optimal Pixel Adjustment Process to suppress first-order histogram flattening.

On the University of Southern California Signal and Image Processing Institute benchmark, this scheme achieves average Peak Signal-to-Noise Ratio (PSNR) values between 40.40 dB and 46.76 dB (outperforming uniform-depth embedding by up to 5.42 dB in PSNR evaluations) while maintaining exact byte-level secret recovery across all tested configurations.

---

## Implemented Algorithms and Comparative Testbed

Beyond the proposed method, this codebase provides a comprehensive reference implementation of state-of-the-art reversible data hiding and steganographic baselines:

- **Proposed Framework:**
  - Standard MLSB substitution (`mlsb.cpp`).
  - Bit-sliced MLSB substitution (`mlsb_bs.cpp`).
  - Shuffled mixed-depth MLSB allocation (`mlsb_mixed.cpp`).
  - Full-range combinatorial MLSB evaluation (`mlsb_all.cpp`).
- **Prediction Error Expansion Family:**
  - Median Edge Detector predictor (`pee_med.cpp`).
  - Adaptive Median Edge Detector expansion (`pee_med_adaptive.cpp`).
  - Diamond prediction context (`pee_diamond.cpp`).
  - Multi-layer Prediction Error Expansion (`pee_multi.cpp`).
  - Checkerboard safe-context expansion (`pee_checker_safe.cpp`).
- **Pixel Value Ordering Family:**
  - Classical Pixel Value Ordering (`pvo.cpp`).
  - Inband Pixel Value Ordering with embedded overhead (`pvo_inband.cpp`).
- **Histogram Shifting & Classical Steganography:**
  - Classical Histogram Shifting (`histogram_shifting.cpp`).
  - Least Significant Bit matching (`lsb_matching.cpp`).
  - Least Significant Bit matching revisited (`lsb_matching_revisited.cpp`).
  - Least Significant Bit Huffman entropy embedding (`lsb_huffman.cpp`).
- **Steganalysis and Metric Verification Suite:**
  - Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (`psnr.cpp`, `ssim.cpp`).
  - Regular-Singular steganalysis (`rs_analysis.cpp`).
  - Sample Pair Analysis (`spa_analysis.cpp`).
  - Weighted Stego-image analysis (`ws_analysis.cpp`).

---

## Repository Structure

```text
secret_sharing_data_hiding_code/
├── finite_field/                           # Galois Field GF(2^8) arithmetic library
│   ├── binary_field.h / .cpp               # Multiplication/inversion tables & Horner evaluation
│   ├── prime_field.h / .cpp                # Prime field reference utilities
│   └── finite_field.vcxproj                # Finite field Visual Studio project
├── secret_sharing_data_hiding/             # Main algorithmic engine and testbed
│   ├── secret_sharing.cpp                  # CLI entry point, batch tests, and visual attacks
│   ├── mlsb.h / .cpp                       # Base Modified Least-Significant-Bit implementation
│   ├── mlsb_mixed.h / .cpp                 # Shuffled mixed-depth MLSB scheme
│   ├── mlsb_bs.h / .cpp                    # Bit-sliced MLSB implementation
│   ├── pee_med.h / .cpp                    # Prediction Error Expansion using Median Edge Detector
│   ├── pee_diamond.h / .cpp                # Prediction Error Expansion using diamond context
│   ├── pvo.h / .cpp                        # Pixel Value Ordering implementation
│   ├── pvo_inband.h / .cpp                 # Inband Pixel Value Ordering
│   ├── histogram_shifting.h / .cpp         # Histogram Shifting reversible data hiding
│   ├── lsb_matching.h / .cpp               # Least Significant Bit matching steganography
│   └── secret_sharing_data_hiding.vcxproj  # Core Visual Studio project
├── tiff_io/                                # High-precision image I/O and steganalysis
│   ├── tiff_io.h / .cpp                    # Grayscale Tagged Image File Format reader/writer
│   ├── psnr.cpp / ssim.cpp                 # Image fidelity metrics
│   ├── rs_analysis.cpp                     # Regular-Singular steganalysis
│   ├── spa_analysis.cpp                    # Sample Pair Analysis
│   ├── ws_analysis.cpp                     # Weighted Stego-image analysis
│   └── tiff_io.vcxproj                     # Image I/O Visual Studio project
├── libtiff/                                # LibTIFF development headers and static libraries
└── secret_sharing_data_hiding.sln          # Master Visual Studio solution
```

---

## Build Prerequisites

- **Compiler:** C++17 compliant compiler (Microsoft Visual C++ 2019/2022, GCC 9+, or Clang 10+).
- **Libraries:** LibTIFF (pre-bundled in `libtiff/`).

---

## Building and Running

### Building with Microsoft Visual Studio
1. Open `secret_sharing_data_hiding.sln` in Visual Studio.
2. Select configuration **Release** and platform **x64**.
3. Build the solution. The binary is written to `x64/Release/`.

### Command-Line Execution Options
The main executable supports specific operational modes via command-line arguments:

```bash
# 1. Run the comprehensive benchmark test suite
# (Evaluates capacity curves, steganalysis, and comparative baselines)
./secret_sharing_data_hiding

# 2. Run with shuffled mixed-depth mode explicitly enabled
./secret_sharing_data_hiding --shuffled

# 3. Generate visual attack figures on benchmark images (e.g., jetplane)
./secret_sharing_data_hiding --visual-attack-only
```

---

## Citation

If you use this benchmark testbed or the proposed mixed-depth algorithms in your research, please cite:

```bibtex
@article{lin2026lossless,
  author    = {Lin, Li-Chiun and Yang, Cheng-Hsing and Huang, Cheng-Ta and Wang, Shiuh-Jeng},
  title     = {Lossless Meaningful Secret Image Sharing via Shuffled Mixed-Depth Modified Least-Significant-Bit Substitution},
  journal   = {Journal of Information Security and Applications},
  year      = {2026},
  note      = {Under review}
}
```
