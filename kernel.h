#ifndef KERNEL_H
#define KERNEL_H

#include <stdint.h>
#include <cuda_runtime.h> // Für dim3
#include "config.h"       // Für FoundResult etc.

// Kernel zur Rekonstruktion und Überprüfung der Schlüssel
extern "C" __global__ void reconstruct_kernel(
    uint32_t start_timestamp,      // Startzeit des Zieltages (Unix UTC Sek.)
    uint32_t total_seconds,        // Anzahl Sekunden (immer 86400 für einen Tag)
    uint32_t pid_min,              // Start-PID
    uint32_t pid_max,              // End-PID (inklusiv)
    uint32_t brute_force_range,    // Anzahl Brute-Force-Variationen (1 falls keine, sonst 2^N)
    const uint8_t* __restrict__ d_targets, // Device-Pointer zu flachen Ziel-HASH160s
    int num_targets,               // Anzahl der Ziel-Hashes
    FoundResult* __restrict__ d_results,   // Device-Pointer zum Ergebnis-Buffer
    unsigned int* __restrict__ d_found_count, // Device-Pointer zum Zähler gefundener Keys
    unsigned int max_results       // Maximale Anzahl speicherbarer Ergebnisse
);

#endif // KERNEL_H