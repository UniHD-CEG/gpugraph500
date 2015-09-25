cmake_minimum_required(VERSION 2.8)
project(BFS_Multinode)

# This is a built-in utility to grab all the source files from a specific
# location into a list
AUX_SOURCE_DIRECTORY(. LOG_SOURCES)

# This is all you need to do to tell CMake that a specific list of files
# should be built as a library (shared in this case)
add_library(Logging SHARED ${LOG_SOURCES})
