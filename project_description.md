# Project Description

## CUDA-Based Signal Processing: CPU vs GPU Performance Comparison

I developed a signal-processing capstone to apply CUDA to a concrete workload and explore how GPU acceleration behaves once data movement is included. The program generates reproducible noisy signals containing 50 Hz, 200 Hz, and 500 Hz components, then applies a 250 Hz low-pass filter using FFTW on the CPU and batched cuFFT plus CUDA kernels on the GPU.

I evaluated workloads of 256 and 1,024 signals, each with 4,096 samples. In the recorded results, GPU compute time was 1.74906 ms and 6.7167 ms, corresponding to 4.36x and 4.90x compute speedups over the CPU. End-to-end GPU time, including transfers, was 9.50064 ms and 37.8017 ms, slower than CPU time for these measurements. This made the transfer overhead an important part of the result rather than hiding it behind the faster kernel/FFT computation.

The recorded CPU and GPU output comparison reports zero error at the eight-decimal-place CSV precision. The repository includes the code, build/run script, result tables, and plots. Generated signal and output CSVs are recreated by the run script because the largest files exceed GitHub's per-file upload limit. The original archive identifies Google Colab but does not specify its exact GPU model or CUDA version, so those details should be captured when reproducing the run.

The main lessons were how to use cuFFT batched transforms, allocate and transfer device data, write CUDA kernels for frequency filtering and normalization, and interpret compute-only versus end-to-end timings. This project demonstrates that GPU speedup depends on the whole pipeline and workload, not just the speed of device computation.

