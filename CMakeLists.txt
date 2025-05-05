# CMake Mindestversion
cmake_minimum_required(VERSION 3.18)

# Projektname
project(BitcoinXPCollider LANGUAGES CXX CUDA)

# --- Setze KEINE globalen CUDA flags hier ---

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Finde das CUDA Toolkit
find_package(CUDA REQUIRED)

# Finde Threads
find_package(Threads REQUIRED)

# Include Verzeichnisse
include_directories(include)

# --- CUDA Architektur ---
# Manuell für GTX 660 (Kepler)
set(CMAKE_CUDA_ARCHITECTURES 35)
message(STATUS "Building for CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")


# --- Quelldateien ---
# main.cu wird *nicht* mehr Teil der Bibliothek
set(CUDA_LIB_SOURCES
    src/kernel.cu
    src/ecc_cuda.cu
    src/hash_cuda.cu
    src/utils.cu
    # src/main.cu # <-- HIER ENTFERNT
)

# --- Erstelle eine STATISCHE Bibliothek mit den CUDA Helfer-Quellen ---
add_library(btc_xp_collider_lib STATIC
    ${CUDA_LIB_SOURCES} # <-- Geänderte Variable verwenden
)

# --- Aktiviere separate Kompilierung (RDC) für die Bibliothek ---
set_target_properties(btc_xp_collider_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# --- Erstelle die finale Executable ---
# main.cu ist die Hauptquelldatei für die Executable
add_executable(btc_xp_collider src/main.cu) # <-- main.cu hier belassen

# --- Linke die Executable gegen die CUDA Bibliothek und Threads ---
target_link_libraries(btc_xp_collider PRIVATE
    btc_xp_collider_lib     # Unsere CUDA Lib mit RDC
    Threads::Threads        # Pthread Lib
)

# ... (Rest der Datei) ...
# --- Hilfsziele (unverändert) ---
add_custom_target(create_dirs
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/targets
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/output
    COMMENT "Creating output and targets directories"
)

# Meldungen (unverändert)
message(STATUS "Konfiguration abgeschlossen. Führen Sie 'make' oder 'cmake --build .' aus.")
message(STATUS "Legen Sie Ihre Zieldatei (z.B. btc.bin) in das 'targets'-Verzeichnis im Build-Ordner.")
message(STATUS "Ausgabe wird in 'output/found.txt' im Build-Ordner gespeichert.")