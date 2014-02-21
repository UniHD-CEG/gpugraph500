TEMPLATE = app
CONFIG += console
#CONFIG -= app_bundle
CONFIG -= qt core
CONFIG += no_autoqmake
#CONFIG += debug

SOURCES += main.cpp \
    distmatrix2d.cpp \
    generator/utils.c \
    generator/splittable_mrg.c \
    generator/make_graph.c \
    generator/graph_generator.c \
    globalbfs.cpp \
    simplecpubfs.cpp \
    validate/onesided.c \
    validate/onesided_emul.c \
    validate/validate.cpp

OTHER_FILES += \
    generator/README \
    generator/LICENSE_1_0.txt \
    generator/generator.includes \
    generator/generator.files \
    generator/generator.creator.user \
    generator/generator.creator \
    generator/generator.config \
    keeneland.sh

HEADERS += distmatrix2d.h \
    generator/utils.h \
    generator/user_settings.h \
    generator/splittable_mrg.h \
    generator/mod_arith_64bit.h \
    generator/mod_arith_32bit.h \
    generator/mod_arith.h \
    generator/make_graph.h \
    generator/graph_generator.h \
    globalbfs.h \
    simplecpubfs.h \
    validate/validate.h \
    validate/onesided.h \
    validate/mpi_workarounds.h

opencl{

SOURCES +=  \
    opencl/opencl_bfs.cpp

HEADERS += \
    opencl/opencl_bfs.h \
    opencl/OCLrunner.hh

LIBS += -lOpenCL

}

gunrock{

SOURCES += \
    gunrock/gunrockbfs.cpp

HEADERS += \
    gunrock/gunrockbfs.h
}

# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc
QMAKE_CC_RELEASE = $$QMAKE_CC
QMAKE_CC_DEBUG = $$QMAKE_CC

#QMAKE_CFLAGS += $$system(mpicc --showme:compile)
QMAKE_CFLAGS_RELEASE += -O3
QMAKE_CFLAGS_DEBUG += -O0
#QMAKE_LFLAGS += $$system(mpicxx --showme:link)
#QMAKE_CXXFLAGS += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
QMAKE_CXXFLAGS_RELEASE += -O3
QMAKE_CXXFLAGS_DEBUG += -O0
QMAKE_CFLAGS += -std=c99 -fopenmp
#QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
