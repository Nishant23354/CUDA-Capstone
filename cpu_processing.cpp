
#include <fftw3.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {

    int num_signals = 256;
    int num_samples = 4096;
    double sample_rate = 2048.0;
    double cutoff_frequency = 250.0;

    std::string input_file =
        "data/input_signals.csv";

    std::string output_file =
        "output/cpu_output.csv";

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

    // Read input signals
    std::ifstream input(input_file);

    if (!input.is_open()) {
        std::cerr << "Cannot open input file.\n";
        return 1;
    }

    std::vector<double> signals(
        static_cast<size_t>(num_signals) * num_samples);

    std::string line;
    std::getline(input, line); // Skip CSV header

    int count = 0;

    while (std::getline(input, line)) {

        if (count >= num_signals * num_samples) break;

        std::stringstream ss(line);
        std::string field;

        // Skip signal_id, sample_index and time
        std::getline(ss, field, ',');
        std::getline(ss, field, ',');
        std::getline(ss, field, ',');

        if (!std::getline(ss, field, ',')) {
            std::cerr << "Invalid CSV row.\n";
            return 1;
        }

        signals[count++] = std::stod(field);
    }

    input.close();

    if (count != num_signals * num_samples) {
        std::cerr << "Input data size mismatch.\n";
        return 1;
    }

    // Allocate FFTW memory
    double* fft_input =
        fftw_alloc_real(num_samples);

    fftw_complex* frequency_data =
        fftw_alloc_complex(num_samples / 2 + 1);

    fftw_plan forward_plan = fftw_plan_dft_r2c_1d(
        num_samples, fft_input, frequency_data, FFTW_ESTIMATE);

    fftw_plan inverse_plan = fftw_plan_dft_c2r_1d(
        num_samples, frequency_data, fft_input, FFTW_ESTIMATE);

    if (!fft_input || !frequency_data ||
        !forward_plan || !inverse_plan) {
        std::cerr << "FFTW allocation or plan creation failed.\n";
        return 1;
    }

    std::vector<double> output(
        static_cast<size_t>(num_signals) * num_samples);

    // Measure FFT processing time
    auto start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < num_signals; s++) {

        for (int i = 0; i < num_samples; i++) {
            fft_input[i] =
                signals[static_cast<size_t>(s) * num_samples + i];
        }

        // Forward FFT
        fftw_execute(forward_plan);

        // Low-pass filter
        for (int k = 0; k <= num_samples / 2; k++) {

            double frequency =
                static_cast<double>(k) * sample_rate / num_samples;

            if (frequency > cutoff_frequency) {
                frequency_data[k][0] = 0.0;
                frequency_data[k][1] = 0.0;
            }
        }

        // Inverse FFT
        fftw_execute(inverse_plan);

        // Normalize inverse FFT output
        for (int i = 0; i < num_samples; i++) {
            output[static_cast<size_t>(s) * num_samples + i] =
                fft_input[i] / num_samples;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double processing_time =
        std::chrono::duration<double, std::milli>(
            end - start).count();

    // Save filtered signals
    std::ofstream out(output_file);

    if (!out.is_open()) {
        std::cerr << "Cannot create output file.\n";
        return 1;
    }

    out << "signal_id,sample_index,time,value\n";
    out << std::fixed << std::setprecision(8);

    for (int s = 0; s < num_signals; s++) {
        for (int i = 0; i < num_samples; i++) {

            out << s << ","
                << i << ","
                << static_cast<double>(i) / sample_rate << ","
                << output[static_cast<size_t>(s) * num_samples + i]
                << "\n";
        }
    }

    out.close();

    std::cout << "\nCPU Processing Completed!\n";
    std::cout << "Number of signals: " << num_signals << "\n";
    std::cout << "Samples per signal: " << num_samples << "\n";
    std::cout << "Cutoff frequency: "
              << cutoff_frequency << " Hz\n";
    std::cout << "Processing time: "
              << processing_time << " ms\n";
    std::cout << "Output file: " << output_file << "\n";

    // Cleanup
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(fft_input);
    fftw_free(frequency_data);

    return 0;
}


