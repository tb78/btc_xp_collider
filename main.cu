#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>      // Für std::put_time, std::setprecision
#include <stdexcept>    // Für Exceptions
#include <limits>       // Für numeric_limits
#include <cmath>        // Für log2
#include <sstream>      // Für String-Parsing

#include "config.h"
#include "utils.h"
#include "kernel.h" // Kernel-Deklaration

// --- Hilfsfunktion für Argument Parsing ---
void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " --date <YYYY-MM-DD> [OPTIONS]\n"
              << "Required:\n"
              << "  --date       <YYYY-MM-DD>          Target date to scan (UTC)\n"
              << "Options:\n"
              << "  --pid-start  <number>              Start Process ID (default: 1000)\n"
              << "  --pid-end    <number>              End Process ID (inclusive, default: 65535)\n"
              << "  --brute-force-bits <N>            Enable N bits of generic brute-force (0-31, default: 0)\n"
              << "  --targets    <path/to/btc.bin>     Path to binary HASH160 target file (default: targets/btc.bin)\n"
              << "  --output     <path/to/found.txt>   Path to output file for results (default: output/found.txt)\n"
              << "  --device     <gpu_id>              GPU device ID to use (default: 0)\n"
              << "  --performance <gkeys_sec>          Optional: Estimated total cluster performance (GKeys/sec) for time estimation\n"
              << "  -h, --help                         Show this help message\n";
}

// Funktion zum Parsen von uint32_t mit Fehlerbehandlung
bool parse_uint32(const char* str, uint32_t& value) {
    try {
        unsigned long long ull_val = std::stoull(str);
        if (ull_val > std::numeric_limits<uint32_t>::max()) {
            std::cerr << "Error: Value " << str << " exceeds uint32_t limit." << std::endl;
            return false;
        }
        value = static_cast<uint32_t>(ull_val);
        return true;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Error: Invalid number format: " << str << std::endl;
        return false;
    } catch (const std::out_of_range& oor) {
        std::cerr << "Error: Number out of range: " << str << std::endl;
        return false;
    }
}


int main(int argc, char** argv) {
    // --- Standardwerte für Parameter ---
    std::string date_str = ""; // Kein Default, muss angegeben werden
    uint32_t pid_start = 1000;
    uint32_t pid_end = 65535;
    uint32_t brute_force_bits = 0;
    std::string target_file = "targets/btc.bin";
    std::string output_file = "output/found.txt";
    int device_id = 0;
    double estimated_performance_gkeys_sec = 0.0; // Für Zeitschätzung

    // --- Argument Parsing ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--date") {
            if (i + 1 < argc) { date_str = argv[++i]; } else { std::cerr << "--date requires an argument (YYYY-MM-DD)." << std::endl; return 1; }
        } else if (arg == "--pid-start") {
            if (i + 1 < argc) { if (!parse_uint32(argv[++i], pid_start)) return 1; } else { std::cerr << "--pid-start requires a number." << std::endl; return 1; }
        } else if (arg == "--pid-end") {
            if (i + 1 < argc) { if (!parse_uint32(argv[++i], pid_end)) return 1; } else { std::cerr << "--pid-end requires a number." << std::endl; return 1; }
        } else if (arg == "--brute-force-bits") {
            if (i + 1 < argc) { if (!parse_uint32(argv[++i], brute_force_bits)) return 1; } else { std::cerr << "--brute-force-bits requires a number (0-31)." << std::endl; return 1; }
        } else if (arg == "--targets") {
            if (i + 1 < argc) { target_file = argv[++i]; } else { std::cerr << "--targets requires a path." << std::endl; return 1; }
        } else if (arg == "--output") {
            if (i + 1 < argc) { output_file = argv[++i]; } else { std::cerr << "--output requires a path." << std::endl; return 1; }
        } else if (arg == "--device") {
             if (i + 1 < argc) { try { device_id = std::stoi(argv[++i]); } catch(...) { std::cerr << "Error parsing device ID." << std::endl; return 1;} } else { std::cerr << "--device requires an argument." << std::endl; return 1; }
        } else if (arg == "--performance") {
             if (i + 1 < argc) { try { estimated_performance_gkeys_sec = std::stod(argv[++i]); } catch(...) { std::cerr << "Error parsing performance value." << std::endl; return 1;} } else { std::cerr << "--performance requires a GKeys/sec value." << std::endl; return 1; }
        }
         else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    // --- 1. Konfiguration validieren und Zeitstempel berechnen ---
    std::cout << "Starting Bitcoin XP Collider..." << std::endl;
    if (date_str.empty()) {
        std::cerr << "Error: --date parameter is required." << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    if (pid_start > pid_end) {
        std::cerr << "Error: PID End must be greater than or equal to PID Start." << std::endl;
        return 1;
    }
     if (brute_force_bits > 31) { // 32 würde 2^32 bedeuten, was 0 in uint32_t ist oder Überlauf bei 1U << 32
        std::cerr << "Error: --brute-force-bits cannot exceed 31." << std::endl;
        return 1;
    }


    std::string start_date_str = date_str + " 00:00:00";
    std::string end_date_str = date_str + " 23:59:59";
    time_t start_timestamp_t, end_timestamp_t;
    try {
        start_timestamp_t = date_to_timestamp(start_date_str);
        end_timestamp_t = date_to_timestamp(end_date_str);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error converting date: " << e.what() << std::endl;
        return 1;
    }
    // Überprüfung auf Gültigkeit (sollte für korrektes Datum immer passen)
     if (start_timestamp_t < 0 || end_timestamp_t < 0 || start_timestamp_t > end_timestamp_t) {
         std::cerr << "Error: Invalid date or timestamp conversion resulted in invalid range." << std::endl;
         return 1;
    }

    uint32_t ts_start = static_cast<uint32_t>(start_timestamp_t);
    //uint32_t ts_end = static_cast<uint32_t>(end_timestamp_t); // Nicht mehr direkt gebraucht
    uint32_t total_seconds = 86400; // Ein ganzer Tag


    // Berechne Brute-Force-Bereich
    uint32_t brute_force_range = 1; // 1 bedeutet keine Brute-Force-Schleife
    if (brute_force_bits > 0) {
        brute_force_range = 1U << brute_force_bits;
    }


    std::cout << "---------------- Configuration ----------------" << std::endl;
    std::cout << "Target Date: " << date_str << " (UTC)" << std::endl;
    std::cout << "Time Range : " << start_date_str << " to " << end_date_str << std::endl;
    std::cout << "             (" << ts_start << ", covering " << total_seconds << " seconds)" << std::endl;
    std::cout << "PID Range  : " << pid_start << " to " << pid_end << " (inclusive)" << std::endl;
    std::cout << "Brute-Force: " << brute_force_bits << " bits (" << brute_force_range << " variations)" << std::endl;
    std::cout << "Target File: " << target_file << std::endl;
    std::cout << "Output File: " << output_file << std::endl;
    std::cout << "GPU Device : " << device_id << std::endl;
    if (estimated_performance_gkeys_sec > 0) {
         std::cout << "Est. Perf. : " << estimated_performance_gkeys_sec << " GKeys/sec (total)" << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;


    // --- 2. Ziel-Hashes laden ---
    std::vector<std::array<uint8_t, HASH160_LENGTH>> h_targets = load_targets(target_file);
    if (h_targets.empty()) {
        std::cerr << "Error: No target hashes loaded from " << target_file << "." << std::endl;
        return 1;
    }
    int num_targets = static_cast<int>(h_targets.size());

    // --- 3. GPU vorbereiten und Speicher allozieren ---
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device_id));
    std::cout << "Using GPU: " << deviceProp.name << std::endl;

    uint8_t* d_targets = nullptr;
    FoundResult* d_results = nullptr;
    unsigned int* d_found_count = nullptr;
    unsigned int h_found_count = 0; // Host-Kopie des Zählers

    size_t targets_size_bytes = (size_t)num_targets * HASH160_LENGTH * sizeof(uint8_t);
    size_t results_size_bytes = (size_t)MAX_FOUND_RESULTS * sizeof(FoundResult);

    CUDA_CHECK(cudaMalloc(&d_targets, targets_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_results, results_size_bytes));
    CUDA_CHECK(cudaMalloc(&d_found_count, sizeof(unsigned int)));

    // --- 4. Daten auf GPU kopieren ---
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets[0].data(), targets_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found_count, 0, sizeof(unsigned int)));

    // --- 5. Kernel Launch Parameter berechnen ---
    unsigned long long pid_range_size = (unsigned long long)pid_end - pid_start + 1;
    unsigned long long total_ticks_per_pid = (unsigned long long)total_seconds * TICKS_PER_SEC;
    unsigned long long base_combinations = pid_range_size * total_ticks_per_pid;
    // Prüfe auf Überlauf bevor Multiplikation mit brute_force_range
    if (pid_range_size > 0 && total_ticks_per_pid > std::numeric_limits<unsigned long long>::max() / pid_range_size) {
        std::cerr << "Error: Base combination count overflows unsigned long long." << std::endl;
        // Aufräumen...
        return 1;
    }

    unsigned long long total_combinations = base_combinations;
    // Prüfe auf Überlauf bei Multiplikation
     if (brute_force_range > 1 && base_combinations > std::numeric_limits<unsigned long long>::max() / brute_force_range) {
          std::cerr << "Error: Total combination count (with brute-force) overflows unsigned long long." << std::endl;
         // Aufräumen...
        return 1;
     }
     total_combinations = base_combinations * brute_force_range;


    if (total_combinations == 0) {
         std::cerr << "Error: Zero combinations to test. Check PID ranges." << std::endl;
         // Aufräumen vor Abbruch
         cudaFree(d_targets);
         cudaFree(d_results);
         cudaFree(d_found_count);
         return 1;
    }

    // Berechne und gib Gesamt-Bitbereich aus
    double total_bits = (total_combinations > 0) ? log2((double)total_combinations) : 0;
    std::cout << "Total combinations to check: " << total_combinations
              << " (~2^" << std::fixed << std::setprecision(1) << total_bits << ")" << std::endl;

    // Geschätzte Zeit berechnen
    if (estimated_performance_gkeys_sec > 0) {
        double total_performance_keys_sec = estimated_performance_gkeys_sec * 1e9;
        if (total_performance_keys_sec > 0) {
             double estimated_seconds = (double)total_combinations / total_performance_keys_sec;
             std::cout << "Estimated time: ";
             if (estimated_seconds < 60.0) {
                 std::cout << std::fixed << std::setprecision(1) << estimated_seconds << " seconds" << std::endl;
             } else if (estimated_seconds < 3600.0) {
                 std::cout << std::fixed << std::setprecision(1) << estimated_seconds / 60.0 << " minutes" << std::endl;
             } else if (estimated_seconds < 86400.0) {
                  std::cout << std::fixed << std::setprecision(1) << estimated_seconds / 3600.0 << " hours" << std::endl;
             } else {
                  std::cout << std::fixed << std::setprecision(1) << estimated_seconds / 86400.0 << " days" << std::endl;
             }
        }
    }


    // Berechne Grid-Größe
    dim3 threads(THREADS_PER_BLOCK, 1, 1);
    unsigned long long num_blocks_ll = (total_combinations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Prüfe, ob Grid-Größe die maximalen Dimensionen überschreitet
    if (num_blocks_ll > (unsigned long long)deviceProp.maxGridSize[0] * deviceProp.maxGridSize[1] * deviceProp.maxGridSize[2] ||
        num_blocks_ll > (unsigned long long)std::numeric_limits<unsigned int>::max())
    {
         std::cerr << "Error: Required grid size (" << num_blocks_ll << ") exceeds CUDA limits." << std::endl;
         // Aufräumen...
         return 1;
    }
     if (num_blocks_ll > (unsigned long long)deviceProp.maxGridSize[0]) {
         std::cerr << "Warning: Required grid size (" << num_blocks_ll << ") exceeds maximum X dimension ("
                   << deviceProp.maxGridSize[0] << "). Performance might be affected." << std::endl;
     }

    dim3 grid((unsigned int)num_blocks_ll, 1, 1);

    std::cout << "Kernel Launch Config: Grid(" << grid.x << "," << grid.y << "," << grid.z
              << "), Threads(" << threads.x << "," << threads.y << "," << threads.z << ")" << std::endl;

    // --- 6. Kernel starten ---
    std::cout << "Launching kernel..." << std::endl;
    auto kernel_start_time = std::chrono::high_resolution_clock::now();

    reconstruct_kernel<<<grid, threads>>>(
        ts_start,
        total_seconds, // Immer 86400
        pid_start, pid_end,
        brute_force_range, // Übergebe den Bereich
        d_targets,
        num_targets,
        d_results,
        d_found_count,
        MAX_FOUND_RESULTS
    );

    CUDA_CHECK(cudaGetLastError()); // Fehler direkt nach Launch prüfen (asynchron!)
    CUDA_CHECK(cudaDeviceSynchronize()); // Warten bis Kernel fertig ist (synchron!)

    auto kernel_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end_time - kernel_start_time);
    double duration_sec = duration.count() / 1000.0;
    std::cout << "Kernel execution finished in " << duration.count() << " ms (" << std::fixed << std::setprecision(2) << duration_sec << " s)." << std::endl;

    // Performance-Metrik
    if (duration_sec > 0) {
        double keys_per_sec = (double)total_combinations / duration_sec;
        std::cout << "Actual Performance: " << std::fixed << std::setprecision(2) << (keys_per_sec / 1e9) << " GKeys/sec (total)" << std::endl;
        std::cout << "                  (" << std::fixed << std::setprecision(2) << (keys_per_sec / 1e6) << " MKeys/sec)" << std::endl;
    }


    // --- 7. Ergebnisse von GPU kopieren ---
    CUDA_CHECK(cudaMemcpy(&h_found_count, d_found_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<FoundResult> h_results;
    if (h_found_count > 0) {
        unsigned int num_results_to_copy = (h_found_count > MAX_FOUND_RESULTS) ? MAX_FOUND_RESULTS : h_found_count;
        std::cout << "Found " << h_found_count << " potential match(es)! (Buffer limited to " << MAX_FOUND_RESULTS << "). Copying " << num_results_to_copy << " result(s)." << std::endl;

        h_results.resize(num_results_to_copy);
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, num_results_to_copy * sizeof(FoundResult), cudaMemcpyDeviceToHost));

        if (h_found_count > MAX_FOUND_RESULTS) {
             std::cout << "Warning: More results found than buffer could hold (" << h_found_count << " > " << MAX_FOUND_RESULTS << "). Consider increasing MAX_FOUND_RESULTS." << std::endl;
        }
    } else {
        std::cout << "No matches found in the specified range." << std::endl;
    }

    // --- 8. Ergebnisse in Datei schreiben ---
    if (!h_results.empty()) {
        std::cout << "Writing results to " << output_file << "..." << std::endl;
        std::ofstream outfile(output_file, std::ios::app); // Append-Modus
        if (!outfile) {
            std::cerr << "Error: Could not open output file: " << output_file << std::endl;
            // Ergebnisse trotzdem auf Konsole ausgeben
             std::cout << "--- Found Keys (Console Output) ---" << std::endl;
             for (const auto& result : h_results) {
                 // Konvertiere Timestamp zurück zu lesbarem Datum/Zeit
                 time_t result_time_t = static_cast<time_t>(result.timestamp);
                 std::tm* ptm = std::gmtime(&result_time_t); // UTC
                 char buffer[32];
                 std::strftime(buffer, 32, "%Y-%m-%d %H:%M:%S", ptm);

                 std::cout << "PrivKey: " << bytes_to_hex(result.private_key, PRIVATE_KEY_LENGTH)
                           << ", H160: " << bytes_to_hex(result.hash160, HASH160_LENGTH)
                           << ", Time: " << buffer << " (UTC)"
                           << ", PID: " << result.pid
                           << ", TickOff: " << (int)result.tick_offset
                           << ", BFVar: " << result.brute_force_variation << std::endl;
             }
             std::cout << "------------------------------------" << std::endl;

        } else {
            auto now = std::chrono::system_clock::now();
            auto now_c = std::chrono::system_clock::to_time_t(now);
            outfile << "# Results from run started at: " << std::put_time(std::gmtime(&now_c), "%Y-%m-%d %H:%M:%S UTC") << "\n";
            outfile << "# Configuration: Date=" << date_str
                    << ", PID=" << pid_start << "-" << pid_end
                    << ", BF_Bits=" << brute_force_bits << "\n";
            for (const auto& result : h_results) {
                 // Konvertiere Timestamp zurück zu lesbarem Datum/Zeit
                 time_t result_time_t = static_cast<time_t>(result.timestamp);
                 std::tm* ptm = std::gmtime(&result_time_t); // UTC
                 char buffer[32];
                 std::strftime(buffer, 32, "%Y-%m-%d %H:%M:%S", ptm);

                outfile << "PrivKey: " << bytes_to_hex(result.private_key, PRIVATE_KEY_LENGTH)
                        << ", H160: " << bytes_to_hex(result.hash160, HASH160_LENGTH)
                        << ", Time: " << buffer << " (UTC)"
                        << ", PID: " << result.pid
                        << ", TickOff: " << (int)result.tick_offset
                        << ", BFVar: " << result.brute_force_variation << "\n";
            }
            outfile.close();
            std::cout << "Successfully wrote " << h_results.size() << " result(s) to " << output_file << std::endl;
        }
    }

    // --- 9. Aufräumen ---
    std::cout << "Cleaning up GPU memory..." << std::endl;
    cudaFree(d_targets); // Sicherer als CUDA_CHECK, gibt keinen Fehler wenn Pointer schon null ist
    cudaFree(d_results);
    cudaFree(d_found_count);

    std::cout << "Program finished." << std::endl;
    return 0;
}