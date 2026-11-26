
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

constexpr double PI = 3.14159265358979323846;

int main(int argc, char* argv[]) {

    int num_signals = 256;
    int num_samples = 4096;
    double sample_rate = 2048.0;
    double noise_amplitude = 0.2;

    std::string output_file =
        "data/input_signals.csv";

    // Read command-line arguments
    if (argc > 1) num_signals = std::stoi(argv[1]);
    if (argc > 2) num_samples = std::stoi(argv[2]);
    if (argc > 3) sample_rate = std::stod(argv[3]);
    if (argc > 4) noise_amplitude = std::stod(argv[4]);
    if (argc > 5) output_file = argv[5];

    if (num_signals <= 0 || num_samples <= 0 ||
        sample_rate <= 1000 || noise_amplitude < 0) {
        std::cerr << "Invalid input parameters.\n";
        return 1;
    }

    std::ofstream file(output_file);

    if (!file.is_open()) {
        std::cerr << "Error opening output file.\n";
        return 1;
    }

    std::mt19937 generator(42);
    std::normal_distribution<double> noise(0.0, noise_amplitude);

    file << "signal_id,sample_index,time,value\n";
    file << std::fixed << std::setprecision(8);

    for (int s = 0; s < num_signals; s++) {

        for (int i = 0; i < num_samples; i++) {

            double time = i / sample_rate;

            double signal =
                1.0 * sin(2.0 * PI * 50.0 * time) +
                0.6 * sin(2.0 * PI * 200.0 * time) +
                0.3 * sin(2.0 * PI * 500.0 * time);

            double noisy_signal = signal + noise(generator);

            file << s << ","
                 << i << ","
                 << time << ","
                 << noisy_signal << "\n";
        }
    }

    file.close();

    std::cout << "Signal generation completed!\n";
    std::cout << "Number of signals: " << num_signals << "\n";
    std::cout << "Samples per signal: " << num_samples << "\n";
    std::cout << "Total samples: "
              << num_signals * num_samples << "\n";
    std::cout << "Sampling frequency: " << sample_rate << " Hz\n";
    std::cout << "Output file: " << output_file << "\n";

    return 0;
}


