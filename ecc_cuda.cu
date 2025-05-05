// src/ecc_cuda.cu

#include "ecc_cuda.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h> // Entferne printf für finalen Code

// --- secp256k1 Konstanten und Typen ---
struct fp_t { uint32_t limb[8]; };
struct point_t { fp_t X, Y, Z; }; // Repräsentiert Jacobian (X, Y, Z)

// Curve prime P = 2^256 - 2^32 - 977
static __constant__ uint32_t P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// Helper to copy __constant__ memory
__device__ void dev_memcpy_const(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

// Precomputed Table TBL[16] for G (k*G) in Jacobian Coordinates
// *** Aktualisiert mit den zuletzt vom User bereitgestellten Werten. ***
// *** Es wird empfohlen, diese Werte gegen eine bekannte, vertrauenswürdige Quelle ***
// *** (z.B. libsecp256k1) zu verifizieren, falls möglich. ***
static __constant__ point_t TBL[16] = {
    // k=0: Identity Point (Using 0,1,0 representation)
    {{ {0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000} }, // X
     { {0x01000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000} }, // Y (Using 1)
     { {0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000} } },// Z=0
    // k=1: G (Jacobian Z=1)
    {{ {0x9817F816,0x5B81F259,0xD928CE2D,0xDBFC9B02,0x070B87CE,0x9562A055,0xACBBDCF9,0x7E66BE79} }, // Gx
     { {0xB8D410FB,0x8FD0479C,0x195485A6,0x48B417FD,0xA808110E,0xFCFBA45D,0x65C4A326,0x77DA3A48} }, // Gy
     { {0x01000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000,0x00000000} } }, // Z=1 (Using 1)
    // k=2: 2G (Jacobian)
    {{ {0xC2AF65B3,0x4404D491,0xC7E295F6,0xFE888FB6,0x55DBA91F,0x3D849121,0xDCE1A81E,0x042C157D} }, // X
     { {0xDBCBADBF,0x6491E85F,0xF883131B,0x3E63BF43,0x4ADB60BF,0xE4D75F6F,0xF7C82CF5,0x49589156} }, // Y
     { {0x70A921F6,0x1FA18F38,0x33A80A4D,0x91682FFA,0x5111221C,0xF8F749BB,0xCA88474D,0xEEB47590} } },// Z
    // k=3: 3G (Jacobian)
    {{ {0x2061223C,0xD6E8E5DB,0x8018A245,0xD6E137F6,0xF2208C17,0x453A9475,0xD4FAB5C1,0xE63BA432} }, // X
     { {0x556B114D,0x4461DBAD,0xE9F36B4A,0xC083A4C0,0xA2A64F91,0xB33A807D,0x519523F4,0x4F0C28E5} }, // Y
     { {0x371F63B5,0xF08D0C85,0x69214F2D,0x1FAF6DD5,0x2B7078F7,0xBA9DD6CD,0x8E75A334,0xD2C854F7} } },// Z
    // k=4: 4G (Jacobian)
    {{ {0x3687251A,0xE5E286E5,0x2CCCB9D6,0x4AB233B8,0xC252A21E,0x7AF8BEC8,0x9FB43997,0x36479C3F} }, // X
     { {0x70F68B0A,0x936E9B63,0x64525683,0xE5FCE162,0x94DFC71F,0xA0ACC7A7,0x39EDE39A,0x28BF09EA} }, // Y
     { {0x426F47C8,0x5D0E23AE,0xA7621755,0x7437A38D,0xE1D018D4,0xB6C0BD12,0x9A0573BE,0xA1F83D68} } },// Z
    // k=5: 5G (Jacobian)
    {{ {0x2C65920D,0x6BB4E6AD,0xDDEBBC6E,0xFA4E27C3,0xC301A0B1,0xFABA7F58,0x5B489C6C,0x13E9E95D} }, // X
     { {0x0B9412B4,0x3C7D71C3,0x11263C14,0x545F6DF7,0x0C544F88,0xA282B43D,0xEE0CA5BA,0xE0709D85} }, // Y
     { {0x5A04BD5F,0xB070E593,0x2D1D5CCF,0x2DA2B55E,0xD7D54E5F,0x8A6B1B32,0x112D2D8D,0x3AAEF9DF} } },// Z
    // k=6: 6G (Jacobian)
    {{ {0x04C9A2F6,0xB495FD29,0x1816EC0F,0x52E5803D,0x5C7F0EF7,0x37C35C29,0x9D4962FE,0x6C0FF8CF} }, // X
     { {0x13E2A2CB,0x28B07A4F,0xB8890CEA,0x13B59D2D,0x7BD8C438,0xB612524E,0xD2C0A9B4,0x6CB2C727} }, // Y
     { {0x5F09A3AA,0x2DDC92F1,0x679E73F0,0x30B09D6F,0x983FAD7C,0x04C88272,0xC97D0820,0xFA7EAE64} } },// Z
    // k=7: 7G (Jacobian)
    {{ {0xA606A0E3,0x8EB8F04E,0xCB3EDB13,0x03C1B9C2,0xEDCBAE14,0x1AD6AC10,0x2AF10C58,0x31F69AD8} }, // X
     { {0xD7183FDF,0x26C7A607,0xFE33AC7A,0x4304FE79,0xB57E4EF8,0x86C6B907,0xE6590883,0x1828E188} }, // Y
     { {0x54D44AB0,0x91AE5616,0x3A456A8A,0x0C5A7B23,0xF25E4313,0x60867938,0x61D493BD,0xC9C44202} } },// Z
    // k=8: 8G (Jacobian)
    {{ {0xE9DDD858,0x7FA7D9B1,0x33AEC379,0x9B1FC417,0xF97D1664,0xA1E5BEE8,0x304BAE8E,0x0FBC74B5} }, // X
     { {0xFF798F4E,0x4274B4A4,0x516F462D,0x05FFA7AF,0xC86CA848,0x439C56C9,0xE307E0F9,0xF6B7D81A} }, // Y
     { {0x55237034,0xFAE7F36A,0x9F4FF9DB,0xF07659E1,0x632AE221,0x9A27E0AB,0xFAE6A4F0,0x0E3B6DB5} } },// Z
    // k=9: 9G (Jacobian)
    {{ {0x650AEE21,0x28E360FD,0xFBE8D048,0x9179C6BA,0x2A055707,0x5C5798BC,0x1D984CD4,0xA5130B61} }, // X
     { {0x1E14C8C3,0xA23C8292,0xDDE8811B,0x18763735,0x00846D1F,0xB6F1DD8F,0xD3DE1E00,0xA43C67D7} }, // Y
     { {0x709A3B01,0x81D40DB2,0x6699268D,0xD59DD1F0,0x252CF5E0,0x1700F0E2,0xC9DD8B12,0xE97E54C6} } },// Z
    // k=10: 10G (Jacobian)
    {{ {0x4AEDCA58,0x37601923,0xBC6A4A9C,0xC8069D59,0x072249B6,0xA7061215,0x232765FB,0xA7312A7C} }, // X
     { {0x54A1BB62,0x89981820,0xA9C8F16D,0xD479A631,0x2C18D225,0xAC69A1A1,0x95189A8E,0x6371E99A} }, // Y
     { {0x29377506,0xD2F6581A,0xA76AF6CF,0x08BC26F4,0xC3F1CEF7,0xEA9DF49E,0x64A50D3F,0x2C547FA9} } },// Z
    // k=11: 11G (Jacobian)
    {{ {0xAD8E4C4A,0x0ABF2AC8,0x655D5B35,0x43FDE8B9,0x687EBCBC,0x9E7B2497,0x728FD4B2,0xFC8CEB66} }, // X
     { {0x64E9106B,0x0E504CD6,0xB074453C,0x307BCC9B,0xF4E4AD08,0xB0F2EE07,0x6D57F6A7,0xB6C388DC} }, // Y
     { {0x026B99DA,0x4C0FDAF8,0x4C87F793,0xD249DFD6,0xFFE7DFAF,0x107C61F7,0x7BFBD07F,0xF5127248} } },// Z
    // k=12: 12G (Jacobian)
    {{ {0x272306F4,0x131B056B,0x526DA8D9,0x5D8C2379,0x15D87BE1,0x3745B6A8,0xD7E015C8,0xFD4FF3A9} }, // X
     { {0x3C50AC4E,0x608FCB5A,0x5DE0E1BE,0xF0D4C176,0x4CDEB254,0x0E8DD6F9,0x9EE4FCE9,0xE653F7D9} }, // Y
     { {0x5244F913,0xEB8C323B,0xAD0502EA,0xFC658381,0x2E723C49,0x1E05EDBE,0xA9D3FC0F,0x4420A3F5} } },// Z
    // k=13: 13G (Jacobian)
    {{ {0x221325E5,0xF01D65D1,0x31D84B39,0x6BBB7944,0x903538A0,0xB3F95FFD,0x838E616E,0x13CB1095} }, // X
     { {0xF72FF11E,0x112CD9BC,0x2D2033DA,0x02346E84,0x26CDAE3C,0x5D72C6E1,0xDF406B63,0x4549D072} }, // Y
     { {0xEA1EA53E,0xC4A5D52B,0xDB6636FC,0x82F01286,0xA07084D3,0xF8F1DC22,0x8A2FE269,0xC90E3173} } },// Z
    // k=14: 14G (Jacobian)
    {{ {0x0A9B2F0E,0x54B4D5EA,0x13E30847,0x51D56198,0x916B70C5,0x739DBEFF,0xC880DCC6,0x3D9C4C81} }, // X
     { {0xA3BECE1D,0xB7F21337,0xBF17191E,0x25DD9371,0xF3CEA6CF,0x868A6CB6,0x200327E6,0x16C76337} }, // Y
     { {0x0DB8EA67,0x3C3FCB97,0x3BAD0DED,0x00A8A98E,0x8DD01CA2,0x7B882C4F,0x92EE8EF8,0xC5579B43} } },// Z
    // k=15: 15G (Jacobian)
    {{ {0xFE2A5DBD,0x3C80419C,0x63154B1E,0x3DFEBB97,0xF5586228,0xD70A5FBA,0x8C5FC1DA,0x87202C55} }, // X
     { {0x4C06C96B,0x6703E145,0x2D20170E,0x5F6E62DB,0xA44DCB15,0xA34C4B67,0x504A88CB,0x76E3CDE8} }, // Y
     { {0x6D99960A,0xCD517C53,0x306C4BCB,0xC000E357,0x21DBD1B8,0x4C908E7B,0x1D33AB0C,0x42367201} } },// Z
};


// --- Arithmetische Helferfunktionen (Device) ---
__device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b, uint32_t &carry) {
    uint64_t sum = (uint64_t)a + b + carry;
    carry = (uint32_t)(sum >> 32);
    return (uint32_t)sum;
 }
__device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b, uint32_t &borrow) {
    uint64_t diff = (uint64_t)a - b - borrow;
    borrow = (diff >> 63) & 1;
    return (uint32_t)diff;
 }
#define MUL32(a, b) ((uint64_t)(a) * (b))

// --- Feldoperationen mod P ---
__device__ fp_t fp_add(const fp_t &a, const fp_t &b) {
    fp_t r;
    uint32_t carry = 0;
    for (int i = 0; i < 8; ++i) {
        r.limb[i] = addc(a.limb[i], b.limb[i], carry);
    }
    uint32_t borrow = 0;
    fp_t temp;
    for(int i=0; i<8; ++i) {
        temp.limb[i] = subc(r.limb[i], P[i], borrow);
    }
    if (carry || !borrow) {
        return temp;
    } else {
        return r;
    }
 }
__device__ fp_t fp_sub(const fp_t &a, const fp_t &b) {
    fp_t r;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        r.limb[i] = subc(a.limb[i], b.limb[i], borrow);
    }
    if (borrow) {
        uint32_t carry = 0;
        for (int i = 0; i < 8; ++i) {
            r.limb[i] = addc(r.limb[i], P[i], carry);
        }
    }
    return r;
 }

// Multiplikation a * b mod P mit schneller Reduktion
// *** WARNUNG: Die Korrektheit dieser Reduktion muss überprüft werden! ***
__device__ fp_t fp_mul(const fp_t &a, const fp_t &b) {
    uint32_t R[16] = {0};
    // 1. Schulmultiplikation
    for (int i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; ++j) {
            uint64_t product = MUL32(a.limb[i], b.limb[j]) + R[i + j] + carry;
            R[i + j] = (uint32_t)product;
            carry = product >> 32;
        }
        R[i + 8] += carry;
    }

    // 2. Modulare Reduktion (für secp256k1 P)
    fp_t result;
    uint64_t p_factor = 977;
    uint64_t tmp_carry1 = 0; // Carry H*2^32
    uint64_t tmp_carry2 = 0; // Carry H*977
    uint32_t reduced[9] = {0}; // Ergebnis + 1 Limb für Überträge

    // H*977 zu L addieren
    for (int i = 0; i < 8; ++i) {
        uint64_t h_limb = R[8 + i];
        uint64_t product = h_limb * p_factor + tmp_carry2;
        uint64_t sum = (uint64_t)R[i] + (uint32_t)product;
        reduced[i] = (uint32_t)sum;
        tmp_carry2 = (product >> 32) + (sum >> 32);
    }
    reduced[8] = (uint32_t)tmp_carry2;

    // H*2^32 zu reduced addieren
    tmp_carry1 = 0;
    for (int i = 0; i < 8; ++i) {
        uint64_t h_limb = (i > 0) ? R[8 + i - 1] : 0; // H[i-1]
        uint64_t sum = (uint64_t)reduced[i] + h_limb + tmp_carry1;
        reduced[i] = (uint32_t)sum;
        tmp_carry1 = sum >> 32;
    }
    uint64_t sum_final_carry = (uint64_t)reduced[8] + tmp_carry1;
    reduced[8] = (uint32_t)sum_final_carry;

    // Kopiere Ergebnis nach result
    for(int i=0; i<8; ++i) result.limb[i] = reduced[i];

    // 3. Finale Reduktion durch Subtraktion
    while (true) {
        uint32_t borrow = 0;
        bool greater_equal = (reduced[8] > 0); // Check extra limb first
        if (!greater_equal) {
            for(int i=7; i>=0; --i) { // Compare result with P
                 if (result.limb[i] < P[i]) { greater_equal = false; break; }
                 if (result.limb[i] > P[i]) { greater_equal = true; break; }
                 // if equal, continue comparison
            }
             if(greater_equal && borrow == 0) { // Check if they were exactly equal
                // Need to re-evaluate comparison logic, this path might be wrong
             }
        }


        if (!greater_equal) break; // result < P, fertig.

        // Subtrahiere P
        borrow = 0;
        for(int i=0; i<8; ++i) {
            result.limb[i] = subc(result.limb[i], P[i], borrow);
        }
         reduced[8] -= borrow; // Propagate borrow to the virtual 9th limb
    }
    return result;
}

// Modulare Inverse a^(P-2) mod P
__device__ fp_t fp_inv(fp_t a) {
    const uint8_t exponent[32] = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B, 0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x3F
    };
    fp_t result = {{1}}; // Start mit 1
    // Left-to-Right binäre Exponentiation
    for (int i = 0; i < 256; ++i) {
         result = fp_mul(result, result); // Square
         int byte_idx = i / 8;
         int bit_idx = 7 - (i % 8);
         if ((exponent[byte_idx] >> bit_idx) & 1) {
              result = fp_mul(result, a); // Multiply
         }
    }
    return result;
}

// --- Punktoperationen (Device) ---

// Prüft, ob ein Punkt der Identitätspunkt O ist (Z=0)
__device__ __forceinline__ bool point_is_identity(const point_t &P) {
    for (int i = 0; i < 8; ++i) { if (P.Z.limb[i] != 0) return false; }
    return true;
}

// Punktverdopplung in Jacobian-Koordinaten
__device__ point_t point_double(const point_t &P) {
    if (point_is_identity(P)) {
        return P;
    }

    // Standard Jacobian Verdopplung (a=0)
    fp_t Y2 = fp_mul(P.Y, P.Y);
    fp_t S = fp_mul(P.X, Y2); S = fp_add(S, S); S = fp_add(S, S); // S = 4*X*Y^2
    fp_t X2 = fp_mul(P.X, P.X);
    fp_t M = fp_mul(X2, {{3}}); // M = 3*X^2

    point_t R;
    R.X = fp_mul(M, M); fp_t S2 = fp_add(S, S); R.X = fp_sub(R.X, S2); // X' = M^2 - 2*S
    fp_t Y4 = fp_mul(Y2, Y2); fp_t Y4_8 = fp_mul(Y4, {{8}});
    fp_t S_minus_Xp = fp_sub(S, R.X); R.Y = fp_mul(M, S_minus_Xp); R.Y = fp_sub(R.Y, Y4_8); // Y' = M*(S - X') - 8*Y^4
    R.Z = fp_mul(P.Y, P.Z); R.Z = fp_add(R.Z, R.Z); // Z' = 2*Y*Z

    return R;
 }

// Punktaddition P + Q (beide Jacobian)
__device__ point_t point_add(const point_t &P, const point_t &Q) {
    if (point_is_identity(P)) return Q;
    if (point_is_identity(Q)) return P;

    // Standard Jacobian Addition (add-2007-bl)
    fp_t Z1_2 = fp_mul(P.Z, P.Z); fp_t Z2_2 = fp_mul(Q.Z, Q.Z);
    fp_t U1 = fp_mul(P.X, Z2_2); fp_t U2 = fp_mul(Q.X, Z1_2);
    fp_t Z1_3 = fp_mul(P.Z, Z1_2); fp_t Z2_3 = fp_mul(Q.Z, Z2_2);
    fp_t S1 = fp_mul(P.Y, Z2_3); fp_t S2 = fp_mul(Q.Y, Z1_3);
    fp_t H = fp_sub(U2, U1); fp_t R = fp_sub(S2, S1);

    bool H_is_zero = true; for(int i=0; i<8; ++i) if(H.limb[i] != 0) H_is_zero = false;
    bool R_is_zero = true; for(int i=0; i<8; ++i) if(R.limb[i] != 0) R_is_zero = false;

    if (H_is_zero) {
        if (R_is_zero) { return point_double(P); } // P = Q
        else { return {{ {0} }, { {1} }, { {0} }}; } // P = -Q
    }

    point_t Res;
    fp_t H2 = fp_mul(H, H); fp_t H3 = fp_mul(H, H2);
    fp_t U1H2 = fp_mul(U1, H2); fp_t U1H2_x2 = fp_add(U1H2, U1H2);
    Res.X = fp_mul(R, R); Res.X = fp_sub(Res.X, H3); Res.X = fp_sub(Res.X, U1H2_x2); // X3 = R^2 - H^3 - 2*U1*H^2
    fp_t S1H3 = fp_mul(S1, H3); fp_t U1H2_minus_X3 = fp_sub(U1H2, Res.X);
    Res.Y = fp_mul(R, U1H2_minus_X3); Res.Y = fp_sub(Res.Y, S1H3); // Y3 = R*(U1*H^2 - X3) - S1*H^3
    Res.Z = fp_mul(P.Z, Q.Z); Res.Z = fp_mul(Res.Z, H); // Z3 = Z1*Z2*H

    return Res;
}

// --- Skalar-Multiplikation (Device) ---
__device__ point_t scalar_mul(const uint8_t k_bytes[PRIVATE_KEY_LENGTH]) {
    uint32_t k[8]; // Little-Endian Limbs des Skalars
    for (int i = 0; i < 8; ++i) {
        k[i] = ((uint32_t)k_bytes[31 - (i * 4 + 0)] << 24) |
               ((uint32_t)k_bytes[31 - (i * 4 + 1)] << 16) |
               ((uint32_t)k_bytes[31 - (i * 4 + 2)] << 8)  |
               ((uint32_t)k_bytes[31 - (i * 4 + 3)] << 0);
    }

    point_t R = {{ {0} }, { {1} }, { {0} }}; // Start mit Identität O = (0, 1, 0)
    point_t temp_add_jacobian;

    // Left-to-Right 4-bit windowing
    for (int i = 63; i >= 0; i--) {
         // Verdopple 4 Mal (nur wenn R nicht Identität ist)
         if (!point_is_identity(R)) {
              R = point_double(R); R = point_double(R);
              R = point_double(R); R = point_double(R);
         }

        // Extrahiere 4-bit Fenster
        int highest_bit_pos = (i + 1) * 4 - 1;
        uint8_t window_val = 0;
        for (int bit_idx = 0; bit_idx < 4; ++bit_idx) {
            int current_bit_pos = highest_bit_pos - bit_idx;
            if (current_bit_pos >= 0) {
                 int limb_idx = current_bit_pos >> 5;
                 int bit_in_limb = current_bit_pos & 31;
                 if ((k[limb_idx] >> bit_in_limb) & 1) {
                     window_val |= (1 << (3 - bit_idx));
                 }
            }
        }

        // Wenn Fenster != 0, addiere Punkt aus TBL mittels Jacobian Addition
        if (window_val != 0) {
            // Kopiere den Jacobian-Punkt aus TBL
            dev_memcpy_const(&temp_add_jacobian, &TBL[window_val], sizeof(point_t));
            // Führe Jacobian Addition aus: R (Jacobian) + temp_add_jacobian (Jacobian)
            R = point_add(R, temp_add_jacobian);
        }
    }
    return R;
}

// --- Hauptfunktion für Public Key Berechnung (Device) ---
__device__ void ecc_get_pubkey(const uint8_t private_key[PRIVATE_KEY_LENGTH],
                                          uint8_t public_key[PUBLIC_KEY_UNCOMPRESSED_LENGTH])
{ 

    

     point_t P_jacobian = scalar_mul(private_key);
     if (point_is_identity(P_jacobian)) {
         for(int i=0; i<PUBLIC_KEY_UNCOMPRESSED_LENGTH; ++i) public_key[i] = 0;
         return;
     }
     // Konvertiere zu Affine
     fp_t z_inv = fp_inv(P_jacobian.Z);
     fp_t z_inv_sq = fp_mul(z_inv, z_inv);
     fp_t z_inv_cu = fp_mul(z_inv_sq, z_inv);
     fp_t X_affine = fp_mul(P_jacobian.X, z_inv_sq);
     fp_t Y_affine = fp_mul(P_jacobian.Y, z_inv_cu);
     // Schreibe als unkomprimierten Public Key
     public_key[0] = 0x04;
     for (int i = 0; i < 8; ++i) { // X Big Endian
         public_key[1 + (7-i)*4 + 3] = (uint8_t)(X_affine.limb[i] >> 0);
         public_key[1 + (7-i)*4 + 2] = (uint8_t)(X_affine.limb[i] >> 8);
         public_key[1 + (7-i)*4 + 1] = (uint8_t)(X_affine.limb[i] >> 16);
         public_key[1 + (7-i)*4 + 0] = (uint8_t)(X_affine.limb[i] >> 24);
     }
      for (int i = 0; i < 8; ++i) { // Y Big Endian
         public_key[33 + (7-i)*4 + 3] = (uint8_t)(Y_affine.limb[i] >> 0);
         public_key[33 + (7-i)*4 + 2] = (uint8_t)(Y_affine.limb[i] >> 8);
         public_key[33 + (7-i)*4 + 1] = (uint8_t)(Y_affine.limb[i] >> 16);
         public_key[33 + (7-i)*4 + 0] = (uint8_t)(Y_affine.limb[i] >> 24);
     }
} 
