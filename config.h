#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>
#include <iostream> // Für CUDA_CHECK in Host-Code
#include <cuda_runtime.h> // Für cudaError_t etc. für CUDA_CHECK

// --- Größen ---
#define HASH160_LENGTH 20
#define PRIVATE_KEY_LENGTH 32
#define PUBLIC_KEY_UNCOMPRESSED_LENGTH 65
#define SHA256_DIGEST_LENGTH 32
#define SHA1_DIGEST_LENGTH 20
#define SHA1_BLOCK_LENGTH 64

// --- OpenSSL PRNG Simulation ---
#define POLL_BUFFER_SIZE 1024 // Größe des Puffers, der bei RAND_poll gefüllt wird
#define MD_STATE_SIZE SHA1_DIGEST_LENGTH // Größe des internen 'md'-Puffers
#define STATE_BUFFER_SIZE 1023 // Größe des 'state'-Puffers

// --- Zeitliche Auflösung ---
// Windows GetTickCount Auflösung ist ca. 10-16ms. 1000/16 ist ein plausibler Wert.
#define TICKS_PER_SEC 62 // (1000 / 16) abgerundet oder aufgerundet? Sicherer Wert: 62 (~16.1ms) oder 63? Nehmen wir 62.

// --- Kernel Konfiguration ---
#define THREADS_PER_BLOCK 256
// Maximale Anzahl an Ergebnissen, die im GPU-Buffer gespeichert werden
#define MAX_FOUND_RESULTS 100
// BRUTE_FORCE_RANGE wird jetzt dynamisch als Kernel-Argument übergeben

// --- Struktur für gefundene Ergebnisse ---
struct FoundResult {
    uint8_t private_key[PRIVATE_KEY_LENGTH];
    uint8_t hash160[HASH160_LENGTH];
    uint32_t timestamp;             // Zeitstempel des Funds (innerhalb des Tages)
    uint32_t pid;                   // PID des Funds
    uint8_t tick_offset;            // Tick-Offset innerhalb der Sekunde
    uint32_t brute_force_variation; // Welche Brute-Force-Variation erfolgreich war
};

// --- CUDA Fehlerprüfung ---
// Wird in main.cu und potenziell anderen Host-Dateien verwendet
static void HandleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA Error] " << cudaGetErrorString(err)
                  << " in " << file << " at line " << line << std::endl;
        // Beenden oder spezifische Fehlerbehandlung
        // exit(EXIT_FAILURE); // Kann in Produktionscode unerwünscht sein
    }
}
// Makro für einfache Nutzung
#define CUDA_CHECK(err) (HandleCudaError(err, __FILE__, __LINE__))


#endif // CONFIG_H