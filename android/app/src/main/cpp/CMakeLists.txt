# Moteur natif de Mirador SI : llama.cpp + un petit pont JNI pour les modèles ouverts (GGUF).
# Les sources de llama.cpp ne sont pas dans le dépôt : `android/obtenir_llama_cpp.sh` les place
# dans cpp/llama.cpp (même version partout : celle de llama-cpp-python 0.3.36 sur PyPI).
cmake_minimum_required(VERSION 3.22.1)
project(mirador_llm C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LLAMA_SRC)
    set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/llama.cpp)
endif()
if(NOT EXISTS ${LLAMA_SRC}/CMakeLists.txt)
    message(FATAL_ERROR "Sources de llama.cpp absentes (${LLAMA_SRC}) : lancer android/obtenir_llama_cpp.sh")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(ANDROID)
    # Un seul binaire pour tous les téléphones : ggml choisit les instructions du processeur à
    # l'exécution (variantes chargées dynamiquement depuis le dossier des bibliothèques natives).
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
        set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
        set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    endif()
endif()

add_subdirectory(${LLAMA_SRC} llama-build)

add_library(mirador_moteur STATIC moteur.cpp)
target_include_directories(mirador_moteur PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${LLAMA_SRC}/common ${LLAMA_SRC}/include)
target_link_libraries(mirador_moteur PUBLIC llama llama-common)

if(ANDROID)
    add_library(mirador_llm SHARED jni_moteur.cpp)
    target_link_libraries(mirador_llm PRIVATE mirador_moteur android log)
else()
    add_executable(essai_moteur essai/essai.cpp)
    target_link_libraries(essai_moteur PRIVATE mirador_moteur)
endif()
