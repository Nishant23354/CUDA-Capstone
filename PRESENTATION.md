# Capstone Demonstration (5–10 minutes)

## Suggested 7-minute flow

### 0:00–0:45 — Goal

Introduce the project: compare CPU and CUDA GPU implementations of FFT-based low-pass filtering. Explain the motivation: use a signal-processing task to study both GPU parallelism and the cost of moving data to and from the GPU.

### 0:45–1:45 — Signal and algorithm

Show the signal definition in `signal_generator.cpp`: 50 Hz, 200 Hz, and 500 Hz components with seeded Gaussian noise. Explain that a 250 Hz low-pass filter should retain the first two components and attenuate 500 Hz.

### 1:45–3:15 — CPU and GPU implementations

Show `cpu_processing.cpp` using FFTW as the baseline. Then show `gpu_processing.cu`: batched cuFFT, CUDA frequency filtering, inverse transform, normalization, and host-device transfers. Emphasize where the project uses GPU hardware.

### 3:15–4:15 — Build and run

Show the requirements in `README.md`, verify the environment with `nvcc --version` and `nvidia-smi`, then run `./run.sh`. Mention that it builds all three programs, generates both workload sizes, and writes CPU/GPU outputs.

### 4:15–5:45 — Proof and results

Show `results/graphs/signal_comparison.png` and `results/final_performance.csv`. For 256 signals, GPU compute is 1.75 ms versus CPU 7.62 ms (4.36x), but GPU end-to-end is 9.50 ms. For 1,024 signals, compute is 6.72 ms versus CPU 32.90 ms (4.90x), while end-to-end is 37.80 ms. Explain that transfers dominate this recorded workload.

### 5:45–6:30 — Accuracy

Show `results/accuracy_results.csv`. The reported error metrics are zero at the saved CSV precision. Clarify that these are the submitted measurements and results can vary across hardware and CUDA environments.

### 6:30–7:00 — Lessons and close

Summarize learning: batched cuFFT, CUDA kernels, memory management, and measuring the full pipeline. Close with the main observation: GPU compute can be much faster, while transfer overhead can erase that gain for end-to-end execution.

## Live demonstration fallback

If a CUDA GPU is unavailable during the presentation, present the source, committed benchmark CSVs, and graphs as proof of the completed execution. State that the original environment was Google Colab; its exact GPU model was not captured in the supplied project archive.

