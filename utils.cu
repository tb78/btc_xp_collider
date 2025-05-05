#include "utils.h"
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept> // Für runtime_error
#include <ctime>     // Für Datumsumwandlung
#include "config.h" // Für HASH160_LENGTH

// Lädt Ziel-Hash160-Werte aus einer Binärdatei
std::vector<std::array<uint8_t, HASH160_LENGTH>> load_targets(const std::string& path) {
    std::vector<std::array<uint8_t, HASH160_LENGTH>> targets;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Fehler: Konnte Zieldatei nicht oeffnen: " << path << std::endl;
        return targets; // Leere Liste zurückgeben
    }
    // Gehe zum Ende, um Größe zu bestimmen
    in.seekg(0, std::ios::end);
    std::streampos fileSize = in.tellg();
    in.seekg(0, std::ios::beg);

    // Überprüfe, ob Dateigröße ein Vielfaches von HASH160_LENGTH ist
    if (fileSize == 0) {
         std::cerr << "Warnung: Zieldatei '" << path << "' ist leer." << std::endl;
         return targets;
    }
    if (fileSize % HASH160_LENGTH != 0) {
        std::cerr << "Warnung: Dateigroesse von " << path << " (" << fileSize
                  << " Bytes) ist kein Vielfaches von " << HASH160_LENGTH << " Bytes." << std::endl;
        // Optional: Abbruch statt nur Warnung
    }

    size_t num_hashes = fileSize / HASH160_LENGTH;
    targets.reserve(num_hashes); // Speicher reservieren

    std::array<uint8_t, HASH160_LENGTH> buffer;
    while (in.read(reinterpret_cast<char*>(buffer.data()), HASH160_LENGTH)) {
        targets.push_back(buffer);
    }

    // Prüfen, ob nach dem Lesen noch Bytes übrig sind (sollte nicht passieren, wenn Größe Vielfaches ist)
    if (in.gcount() != 0 && in.gcount() < HASH160_LENGTH) {
         std::cerr << "Warnung: Ungueltige Anzahl an Bytes (" << in.gcount() << ") am Ende der Datei " << path << std::endl;
    }

    std::cout << "Info: " << targets.size() << " Ziel-Hashes (" << targets.size() * HASH160_LENGTH << " Bytes) aus " << path << " geladen." << std::endl;
    return targets;
}

// Konvertiert Bytes in einen Hex-String
std::string bytes_to_hex(const uint8_t* data, size_t len) {
    if (!data || len == 0) {
        return "";
    }
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    return ss.str();
}

// Funktion zum Konvertieren von YYYY-MM-DD HH:MM:SS zu Unix Timestamp UTC
time_t date_to_timestamp(const std::string& date_str) {
    std::tm t{};
    std::istringstream ss(date_str);

    // Versuche verschiedene Formate zu parsen
    ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
         // Versuche alternatives Format (z.B. ohne Sekunden) falls nötig
         // ss.clear(); // Fehlerflags zurücksetzen
         // ss.str(date_str); // String neu setzen
         // ss >> std::get_time(&t, "%Y-%m-%d %H:%M");
         // if (ss.fail()) {
             throw std::runtime_error("Failed to parse date string (Format: YYYY-MM-DD HH:MM:SS): " + date_str);
         // }
    }

    // Konvertiere zu UTC Timestamp
    #ifdef _WIN32
       // Windows: _mkgmtime für UTC
       time_t timestamp = _mkgmtime(&t);
       if (timestamp == -1) {
            throw std::runtime_error("Failed to convert date to timestamp (check date validity): " + date_str);
       }
       return timestamp;
    #else
       // Linux/macOS: timegm für UTC (POSIX / GNU Erweiterung)
       time_t timestamp = timegm(&t);
        if (timestamp == -1) {
            throw std::runtime_error("Failed to convert date to timestamp (check date validity): " + date_str);
       }
       return timestamp;
    #endif
}