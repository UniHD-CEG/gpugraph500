TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
../sliced_reduction.hpp

# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc
QMAKE_CC_RELEASE = $$QMAKE_CC
QMAKE_CC_DEBUG = $$QMAKE_CC

QMAKE_CFLAGS_RELEASE += -O3 -march=native -mtune=native #-fast
#QMAKE_CFLAGS_RELEASE += -fast
QMAKE_CFLAGS_DEBUG += -Og -g
QMAKE_CXXFLAGS_RELEASE += -O3 -march=native -mtune=native #-fast
#QMAKE_CXXFLAGS_RELEASE += -fast
QMAKE_CXXFLAGS_DEBUG += -Og -g
QMAKE_CFLAGS += -std=c99 -fopenmp
#QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -fopenmp -DINSTRUMENTED
QMAKE_LFLAGS += -fopenmp
QMAKE_LFLAGS_RELEASE += -Wl,-O3 -march=native -mtune=native #-fast
#QMAKE_LFLAGS_RELEASE += -Wl,-O3
QMAKE_LFLAGS_DEBUG += -Og
