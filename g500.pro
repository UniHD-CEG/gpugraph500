TEMPLATE = app
#CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
#CONFIG += debug

SOURCES += main.cpp \
    distmatrix2d.cpp \
    generator/utils.c \
    generator/splittable_mrg.c \
    generator/make_graph.c \
    generator/graph_generator.c \
    globalbfs.cpp \
    simplecpubfs.cpp

OTHER_FILES += \
    generator/README \
    generator/LICENSE_1_0.txt \
    generator/generator.includes \
    generator/generator.files \
    generator/generator.creator.user \
    generator/generator.creator \
    generator/generator.config

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
    simplecpubfs.h

# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc

QMAKE_CFLAGS += $$system(mpicc --showme:compile)
QMAKE_LFLAGS += $$system(mpicxx --showme:link)
QMAKE_CXXFLAGS += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
QMAKE_CXXFLAGS_RELEASE += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
QMAKE_CFLAGS += -std=c99
#QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp
