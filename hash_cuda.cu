#include "hash_cuda.h"
#include <cuda_runtime.h>

// --- SHA-256 Implementierung ---

// SHA-256 Hilfsfunktionen
__device__ __forceinline__ uint32_t sha256_rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t sha256_shr(uint32_t x, uint32_t n) { return x >> n; }
__device__ __forceinline__ uint32_t sha256_Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t sha256_Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t sha256_Sigma0(uint32_t x) { return sha256_rotr(x, 2) ^ sha256_rotr(x, 13) ^ sha256_rotr(x, 22); }
__device__ __forceinline__ uint32_t sha256_Sigma1(uint32_t x) { return sha256_rotr(x, 6) ^ sha256_rotr(x, 11) ^ sha256_rotr(x, 25); }
__device__ __forceinline__ uint32_t sha256_sigma0(uint32_t x) { return sha256_rotr(x, 7) ^ sha256_rotr(x, 18) ^ sha256_shr(x, 3); }
__device__ __forceinline__ uint32_t sha256_sigma1(uint32_t x) { return sha256_rotr(x, 17) ^ sha256_rotr(x, 19) ^ sha256_shr(x, 10); }

// SHA-256 Konstanten K
__constant__ uint32_t sha256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 Prozessiert einen 512-bit (64 Byte) Block
__device__ void sha256_process_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t W[64];
    // Daten in Big-Endian laden und Schedule berechnen
    for (int t = 0; t < 16; ++t) {
        W[t] = ((uint32_t)block[t * 4 + 0] << 24) | ((uint32_t)block[t * 4 + 1] << 16) |
               ((uint32_t)block[t * 4 + 2] << 8)  | ((uint32_t)block[t * 4 + 3] << 0);
    }
    for (int t = 16; t < 64; ++t) {
        W[t] = sha256_sigma1(W[t - 2]) + W[t - 7] + sha256_sigma0(W[t - 15]) + W[t - 16];
    }

    // Runden
    uint32_t a = state[0]; uint32_t b = state[1]; uint32_t c = state[2]; uint32_t d = state[3];
    uint32_t e = state[4]; uint32_t f = state[5]; uint32_t g = state[6]; uint32_t h = state[7];

    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = h + sha256_Sigma1(e) + sha256_Ch(e, f, g) + sha256_K[t] + W[t];
        uint32_t T2 = sha256_Sigma0(a) + sha256_Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    // Update State
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// SHA-256 für den 65-Byte Public Key
__device__ void sha256_pubkey(const uint8_t pub_key[PUBLIC_KEY_UNCOMPRESSED_LENGTH], uint8_t out_sha256[SHA256_DIGEST_LENGTH]) {
     uint32_t state[8] = { // Initial H Values
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Block 1: Erste 64 Bytes des Public Keys
    sha256_process_block(state, pub_key);

    // Block 2: Letztes Byte + Padding + Länge
    uint8_t last_block[64]; // SHA256_BLOCK_LENGTH = 64
    last_block[0] = pub_key[64]; // Das 65. Byte
    last_block[1] = 0x80;        // Padding-Beginn

    for(int i = 2; i < 56; ++i) { // Sicherer als memset
        last_block[i] = 0;
    }

    uint64_t bit_len = 520; // Längenangabe (65 Bytes * 8 = 520 Bits) als 64-bit Big-Endian
    last_block[56] = (uint8_t)(bit_len >> 56);
    last_block[57] = (uint8_t)(bit_len >> 48);
    last_block[58] = (uint8_t)(bit_len >> 40);
    last_block[59] = (uint8_t)(bit_len >> 32);
    last_block[60] = (uint8_t)(bit_len >> 24);
    last_block[61] = (uint8_t)(bit_len >> 16);
    last_block[62] = (uint8_t)(bit_len >> 8);
    last_block[63] = (uint8_t)(bit_len >> 0);

    sha256_process_block(state, last_block); // Verarbeite den zweiten (letzten) Block

    // Schreibe finalen Hash (State) in Big-Endian
    for (int i = 0; i < 8; ++i) {
        out_sha256[i * 4 + 0] = (uint8_t)(state[i] >> 24);
        out_sha256[i * 4 + 1] = (uint8_t)(state[i] >> 16);
        out_sha256[i * 4 + 2] = (uint8_t)(state[i] >> 8);
        out_sha256[i * 4 + 3] = (uint8_t)(state[i] >> 0);
    }
}

// --- RIPEMD-160 Implementierung ---

// RIPEMD-160 Hilfsfunktionen
__device__ __forceinline__ uint32_t ripemd160_rotl(uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ uint32_t ripemd160_F(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
__device__ __forceinline__ uint32_t ripemd160_G(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ __forceinline__ uint32_t ripemd160_H(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
__device__ __forceinline__ uint32_t ripemd160_I(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
__device__ __forceinline__ uint32_t ripemd160_J(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

// RIPEMD-160 Konstanten
__constant__ uint32_t ripemd160_K[5]  = { 0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E };
__constant__ uint32_t ripemd160_KK[5] = { 0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000 };
__constant__ uint8_t ripemd160_S[5][16] = {
    {11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8}, {7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12},
    {11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5}, {11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12},
    {9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6} };
__constant__ uint8_t ripemd160_SS[5][16] = {
    {8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6}, {9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11},
    {9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5}, {15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8},
    {8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11} };
__constant__ uint8_t ripemd160_R[5][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8},
    {3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12}, {1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2},
    {4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13} };
__constant__ uint8_t ripemd160_RR[5][16] = {
    {5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12}, {6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2},
    {15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13}, {8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14},
    {12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11} };

// RIPEMD-160 Verarbeitet einen 512-bit (64 Byte) Block
__device__ void ripemd160_process_block(uint32_t state[5], const uint8_t block[64]) {
    uint32_t X[16];
    for (int i = 0; i < 16; ++i) { // Daten in Little-Endian laden
        X[i] = ((uint32_t)block[i * 4 + 0] << 0)  | ((uint32_t)block[i * 4 + 1] << 8) |
               ((uint32_t)block[i * 4 + 2] << 16) | ((uint32_t)block[i * 4 + 3] << 24);
    }

    uint32_t A = state[0], B = state[1], C = state[2], D = state[3], E = state[4];
    uint32_t AA = state[0], BB = state[1], CC = state[2], DD = state[3], EE = state[4];
    uint32_t T;

    // Runden (Linke Linie)
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(A + ripemd160_F(B, C, D) + X[ripemd160_R[0][j]] + ripemd160_K[0], ripemd160_S[0][j]) + E; A = E; E = D; D = ripemd160_rotl(C, 10); C = B; B = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(A + ripemd160_G(B, C, D) + X[ripemd160_R[1][j]] + ripemd160_K[1], ripemd160_S[1][j]) + E; A = E; E = D; D = ripemd160_rotl(C, 10); C = B; B = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(A + ripemd160_H(B, C, D) + X[ripemd160_R[2][j]] + ripemd160_K[2], ripemd160_S[2][j]) + E; A = E; E = D; D = ripemd160_rotl(C, 10); C = B; B = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(A + ripemd160_I(B, C, D) + X[ripemd160_R[3][j]] + ripemd160_K[3], ripemd160_S[3][j]) + E; A = E; E = D; D = ripemd160_rotl(C, 10); C = B; B = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(A + ripemd160_J(B, C, D) + X[ripemd160_R[4][j]] + ripemd160_K[4], ripemd160_S[4][j]) + E; A = E; E = D; D = ripemd160_rotl(C, 10); C = B; B = T; }

    // Runden (Rechte Linie)
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(AA + ripemd160_J(BB, CC, DD) + X[ripemd160_RR[0][j]] + ripemd160_KK[0], ripemd160_SS[0][j]) + EE; AA = EE; EE = DD; DD = ripemd160_rotl(CC, 10); CC = BB; BB = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(AA + ripemd160_I(BB, CC, DD) + X[ripemd160_RR[1][j]] + ripemd160_KK[1], ripemd160_SS[1][j]) + EE; AA = EE; EE = DD; DD = ripemd160_rotl(CC, 10); CC = BB; BB = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(AA + ripemd160_H(BB, CC, DD) + X[ripemd160_RR[2][j]] + ripemd160_KK[2], ripemd160_SS[2][j]) + EE; AA = EE; EE = DD; DD = ripemd160_rotl(CC, 10); CC = BB; BB = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(AA + ripemd160_G(BB, CC, DD) + X[ripemd160_RR[3][j]] + ripemd160_KK[3], ripemd160_SS[3][j]) + EE; AA = EE; EE = DD; DD = ripemd160_rotl(CC, 10); CC = BB; BB = T; }
    for (int j = 0; j < 16; ++j) { T = ripemd160_rotl(AA + ripemd160_F(BB, CC, DD) + X[ripemd160_RR[4][j]] + ripemd160_KK[4], ripemd160_SS[4][j]) + EE; AA = EE; EE = DD; DD = ripemd160_rotl(CC, 10); CC = BB; BB = T; }

    // Kombiniere Ergebnisse
    T = state[1] + CC + D;
    state[1] = state[2] + DD + E;
    state[2] = state[3] + EE + A;
    state[3] = state[4] + AA + B;
    state[4] = state[0] + BB + C;
    state[0] = T;
}

// RIPEMD-160 Hauptfunktion für den 32-Byte SHA256-Hash
__device__ void ripemd160_sha256(const uint8_t sha256_hash[SHA256_DIGEST_LENGTH], uint8_t out_ripemd160[HASH160_LENGTH]) {
    uint32_t state[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0}; // Initial State H0-H4
    uint8_t block[64]; // RIPEMD160_BLOCK_LENGTH = 64

    for(int i=0; i<SHA256_DIGEST_LENGTH; ++i) { block[i] = sha256_hash[i]; } // Kopiere den 32-Byte SHA256 Hash
    block[32] = 0x80; // Füge Padding hinzu (0x80)
    for(int i=33; i<56; ++i) { block[i] = 0; } // Fülle mit Nullen bis Byte 55

    uint64_t bit_len = 256; // Füge Längenangabe hinzu (32 Bytes = 256 Bits) als 64-bit Little-Endian
    block[56] = (uint8_t)(bit_len >> 0);
    block[57] = (uint8_t)(bit_len >> 8);
    block[58] = (uint8_t)(bit_len >> 16);
    block[59] = (uint8_t)(bit_len >> 24);
    block[60] = (uint8_t)(bit_len >> 32);
    block[61] = (uint8_t)(bit_len >> 40);
    block[62] = (uint8_t)(bit_len >> 48);
    block[63] = (uint8_t)(bit_len >> 56);

    ripemd160_process_block(state, block); // Verarbeite den Block

    // Schreibe finalen Hash (State) in Little-Endian
    for (int i = 0; i < 5; ++i) {
        out_ripemd160[i * 4 + 0] = (uint8_t)(state[i] >> 0);
        out_ripemd160[i * 4 + 1] = (uint8_t)(state[i] >> 8);
        out_ripemd160[i * 4 + 2] = (uint8_t)(state[i] >> 16);
        out_ripemd160[i * 4 + 3] = (uint8_t)(state[i] >> 24);
    }
}

// --- Kombinierte HASH160 Funktion ---
__device__ void compute_hash160(const uint8_t pub_key[PUBLIC_KEY_UNCOMPRESSED_LENGTH], uint8_t out_hash160[HASH160_LENGTH]) {
    uint8_t sha256_result[SHA256_DIGEST_LENGTH];
    sha256_pubkey(pub_key, sha256_result);
    ripemd160_sha256(sha256_result, out_hash160);
}