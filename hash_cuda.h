#ifndef HASH_CUDA_H
#define HASH_CUDA_H

#include <stdint.h>
#include "config.h"

// Berechnet SHA256 eines 65-Byte Public Keys
__device__ void sha256_pubkey(const uint8_t pub_key[PUBLIC_KEY_UNCOMPRESSED_LENGTH],
                             uint8_t out_sha256[SHA256_DIGEST_LENGTH]);

// Berechnet RIPEMD160 eines 32-Byte SHA256 Hashes
__device__ void ripemd160_sha256(const uint8_t sha256_hash[SHA256_DIGEST_LENGTH],
                                uint8_t out_ripemd160[HASH160_LENGTH]);

// Berechnet HASH160(pubkey) = RIPEMD160(SHA256(pubkey))
__device__ void compute_hash160(const uint8_t pub_key[PUBLIC_KEY_UNCOMPRESSED_LENGTH],
                               uint8_t out_hash160[HASH160_LENGTH]);

#endif // HASH_CUDA_H