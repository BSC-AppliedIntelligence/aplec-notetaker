# Built as one more whisper.cpp example, so it links the same libraries as
# whisper-cli: Dockerfile.worker copies this directory into examples/ and adds
# it to examples/CMakeLists.txt. On the host, scripts/build_whisper_batch.sh
# links it against the existing build instead (no recompile).
set(TARGET whisper-batch)
add_executable(${TARGET} whisper_batch.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common whisper ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
