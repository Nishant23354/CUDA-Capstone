
#include <cuda_runtime.h>
#include <cufft.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                           \
            std::cerr << "CUDA Error: "                       \
                      << cudaGetErrorString(error) << "\n";   \
            return 1;                                         \
        }                                                     \
    } while (0)

// cuFFT error checking
#define CUFFT_CHECK(call)                                     \
    do {                                                      \
        cufftResult error = call;                             \
        if (error != CUFFT_SUCCESS) {                         \
            std::cerr << "cuFFT Error: " << error << "\n";     \
            return 1;                                         \
        }                                                     \
    } while (0)

// Low-pass filtering kernel
__global__ void low_pass_filter(
    cufftDoubleComplex* frequency_data,
    int num_samples,
    int num_signals,
    double sample_rate,
    double cutoff_frequency) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int bins_per_signal = num_samples / 2 + 1;
    int total_bins = num_signals * bins_per_signal;

    if (index < total_bins) {

        int bin = index % bins_per_signal;

        double frequency =
            static_cast<double>(bin) * sample_rate / num_samples;

        if (frequency > cutoff_frequency) {
            frequency_data[index].x = 0.0;
            frequency_data[index].y = 0.0;
        }
    }
}

// Normalize inverse FFT output
__global__ void normalize_output(
    double* output,
    int total_samples,
    int num_samples) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_samples) {
        output[index] /= num_samples;
    }
}

int main(int argc, char* argv[]) {

    int num_signals = 256;
    int num_samples = 4096;
    double sample_rate = 2048.0;
    double cutoff_frequency = 250.0;

    std::string input_file =
        "data/input_signals.csv";

    std::string output_file =
        "output/gpu_output.csv";

    if (argc > 1) num_signals = std::stoi(argv[1]);
    if (argc > 2) num_samples = std::stoi(argv[2]);
    if (argc > 3) cutoff_frequency = std::stod(argv[3]);
    if (argc > 4) input_file = argv[4];
    if (argc > 5) output_file = argv[5];

    if (num_signals <= 0 || num_samples <= 0 ||
        cutoff_frequency <= 0 ||
        cutoff_frequency >= sample_rate / 2.0) {

        std::cerr << "Invalid parameters.\n";
        return 1;
    }

    size_t total_samples =
        static_cast<size_t>(num_signals) * num_samples;

    size_t total_bins =
        static_cast<size_t>(num_signals) * (num_samples / 2 + 1);

    // Read input CSV
    std::ifstream input(input_file);

    if (!input.is_open()) {
        std::cerr << "Cannot open input file.\n";
        return 1;
    }

    std::vector<double> host_input(total_samples);

    std::string line;
    std::getline(input, line);

    size_t count = 0;

    while (std::getline(input, line)) {

        if (count >= total_samples) break;

        std::stringstream ss(line);
        std::string field;

        std::getline(ss, field, ',');
        std::getline(ss, field, ',');
        std::getline(ss, field, ',');

        if (!std::getline(ss, field, ',')) {
            std::cerr << "Invalid CSV row.\n";
            return 1;
        }

        host_input[count++] = std::stod(field);
    }

    input.close();

    if (count != total_samples) {
        std::cerr << "Input data size mismatch.\n";
        return 1;
    }

    // Allocate GPU memory
    double* device_input = nullptr;

    cufftDoubleComplex* device_frequency = nullptr;

    CUDA_CHECK(cudaMalloc(
        &device_input,
        total_samples * sizeof(double)));

    CUDA_CHECK(cudaMalloc(
        &device_frequency,
        total_bins * sizeof(cufftDoubleComplex)));

    // Create forward FFT plan
    cufftHandle plan;
    int n[] = {num_samples};

    CUFFT_CHECK(cufftPlanMany(
        &plan,
        1,
        n,
        nullptr, 1, num_samples,
        nullptr, 1, num_samples / 2 + 1,
        CUFFT_D2Z,
        num_signals));

    // Create inverse FFT plan
    cufftHandle inverse_plan;

    CUFFT_CHECK(cufftPlanMany(
        &inverse_plan,
        1,
        n,
        nullptr, 1, num_samples / 2 + 1,
        nullptr, 1, num_samples,
        CUFFT_Z2D,
        num_signals));

    // Create CUDA timing events
    cudaEvent_t start, stop;
    cudaEvent_t compute_start, compute_stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventCreate(&compute_start));
    CUDA_CHECK(cudaEventCreate(&compute_stop));

    // Start total GPU timing
    CUDA_CHECK(cudaEventRecord(start));

    // Copy input from CPU to GPU
    CUDA_CHECK(cudaMemcpy(
        device_input,
        host_input.data(),
        total_samples * sizeof(double),
        cudaMemcpyHostToDevice));

    // Start computation-only timing
    CUDA_CHECK(cudaEventRecord(compute_start));

    // Forward batched FFT
    CUFFT_CHECK(cufftExecD2Z(
        plan,
        device_input,
        device_frequency));

    // Apply low-pass filtering
    int threads = 256;

    int blocks =
        static_cast<int>((total_bins + threads - 1) / threads);

    low_pass_filter<<<blocks, threads>>>(
        device_frequency,
        num_samples,
        num_signals,
        sample_rate,
        cutoff_frequency);

    CUDA_CHECK(cudaGetLastError());

    // Inverse batched FFT
    CUFFT_CHECK(cufftExecZ2D(
        inverse_plan,
        device_frequency,
        device_input));

    // Normalize inverse FFT output
    int normalize_blocks =
        static_cast<int>((total_samples + threads - 1) / threads);

    normalize_output<<<normalize_blocks, threads>>>(
        device_input,
        static_cast<int>(total_samples),
        num_samples);

    CUDA_CHECK(cudaGetLastError());

    // Stop computation-only timing
    CUDA_CHECK(cudaEventRecord(compute_stop));
    CUDA_CHECK(cudaEventSynchronize(compute_stop));

    float compute_time = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &compute_time,
        compute_start,
        compute_stop));

    // Copy processed output back to CPU
    std::vector<double> host_output(total_samples);

    CUDA_CHECK(cudaMemcpy(
        host_output.data(),
        device_input,
        total_samples * sizeof(double),
        cudaMemcpyDeviceToHost));

    // Stop total GPU timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float processing_time = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &processing_time,
        start,
        stop));

    // Save GPU output CSV
    std::ofstream out(output_file);

    if (!out.is_open()) {
        std::cerr << "Cannot create output file.\n";
        return 1;
    }

    out << "signal_id,sample_index,time,value\n";
    out << std::fixed << std::setprecision(8);

    for (int s = 0; s < num_signals; s++) {

        for (int i = 0; i < num_samples; i++) {

            size_t index =
                static_cast<size_t>(s) * num_samples + i;

            out << s << ","
                << i << ","
                << static_cast<double>(i) / sample_rate << ","
                << host_output[index] << "\n";
        }
    }

    out.close();

    // Display results
    std::cout << "\nGPU Processing Completed!\n";

    std::cout << "Number of signals: "
              << num_signals << "\n";

    std::cout << "Samples per signal: "
              << num_samples << "\n";

    std::cout << "Cutoff frequency: "
              << cutoff_frequency << " Hz\n";

    std::cout << "GPU computation time: "
              << compute_time << " ms\n";

    std::cout << "GPU total time (including memory transfers): "
              << processing_time << " ms\n";

    std::cout << "Output file: "
              << output_file << "\n";

    // Cleanup
    cufftDestroy(plan);
    cufftDestroy(inverse_plan);

    cudaFree(device_input);
    cudaFree(device_frequency);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(compute_start);
    cudaEventDestroy(compute_stop);

    return 0;
}


