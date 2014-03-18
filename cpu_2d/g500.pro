TEMPLATE = app
CONFIG += console
CONFIG -= qt core
CONFIG += no_autoqmake
#CONFIG += debug

SOURCES += main.cpp \
    generator/utils.c \
    generator/splittable_mrg.c \
    generator/make_graph.c \
    generator/graph_generator.c \
    simplecpubfs.cpp \
    validate/onesided.c \
    validate/onesided_emul.c \
    validate/validate.cpp \
    cpubfs_bin.cpp

OTHER_FILES += \
    generator/README \
    generator/LICENSE_1_0.txt \
    generator/generator.includes \
    generator/generator.files \
    generator/generator.creator.user \
    generator/generator.creator \
    generator/generator.config \
    keeneland.sh

HEADERS += \
    generator/utils.h \
    generator/user_settings.h \
    generator/splittable_mrg.h \
    generator/mod_arith_64bit.h \
    generator/mod_arith_32bit.h \
    generator/mod_arith.h \
    generator/make_graph.h \
    generator/graph_generator.h \
    simplecpubfs.h \
    validate/validate.h \
    validate/onesided.h \
    validate/mpi_workarounds.h \
    cpubfs_bin.h \
    distmatrix2d.hh \
    globalbfs.hh

opencl{
QMAKE_CXXFLAGS += -D_OPENCL
QMAKE_CFLAGS   += -D_OPENCL

SOURCES +=  \
    opencl/opencl_bfs.cpp

HEADERS += \
    opencl/opencl_bfs.h \
    opencl/OCLrunner.hh

LIBS += -lOpenCL

}

cuda{
QMAKE_CXXFLAGS += -D_CUDA
QMAKE_CFLAGS   += -D_CUDA

SOURCES += \
    cuda/cuda_bfs.cpp

HEADERS += \
    cuda/cuda_bfs.h
}

# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc
QMAKE_CC_RELEASE = $$QMAKE_CC
QMAKE_CC_DEBUG = $$QMAKE_CC

#QMAKE_CFLAGS_RELEASE += -O3 -march=native -mtune=native #-fast
QMAKE_CFLAGS_RELEASE += -fast
QMAKE_CFLAGS_DEBUG += -Og -g
#QMAKE_CXXFLAGS_RELEASE += -O3 -march=native -mtune=native #-fast
QMAKE_CXXFLAGS_RELEASE += -fast
QMAKE_CXXFLAGS_DEBUG += -Og -g
QMAKE_CFLAGS += -std=c99 -fopenmp
#QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -fopenmp -DINSTRUMENTED
QMAKE_LFLAGS += -fopenmp
#QMAKE_LFLAGS_RELEASE += -Wl,-O3 -march=native -mtune=native #-fast
QMAKE_LFLAGS_RELEASE += -Wl,-O3 -fast
QMAKE_LFLAGS_DEBUG += -Og
