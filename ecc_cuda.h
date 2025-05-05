#ifndef ECC_CUDA_H
#define ECC_CUDA_H

#include <stdint.h>
#include "config.h" // Für Key-Längen

// Berechnet den unkomprimierten Public Key aus einem Private Key
__device__ void ecc_get_pubkey(const uint8_t private_key[PRIVATE_KEY_LENGTH],
                              uint8_t public_key[PUBLIC_KEY_UNCOMPRESSED_LENGTH]);

#endif // ECC_CUDA_H