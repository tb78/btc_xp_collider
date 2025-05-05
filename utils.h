#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include "config.h" // Für HASH160_LENGTH

// Lädt Ziel-Hash160-Werte aus einer Binärdatei
std::vector<std::array<uint8_t, HASH160_LENGTH>> load_targets(const std::string& path);

// Konvertiert Bytes (z.B. Private Key, Hash160) in einen Hex-String
std::string bytes_to_hex(const uint8_t* data, size_t len);

// Konvertiert Datumsstring (YYYY-MM-DD HH:MM:SS) in Unix Timestamp (UTC)
time_t date_to_timestamp(const std::string& date_str);

#endif // UTILS_H