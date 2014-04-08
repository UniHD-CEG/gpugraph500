TEMPLATE = app
CONFIG += console
CONFIG -= qt core
CONFIG += no_autoqmake cuda
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
    keeneland.sh \
    cuda/cuda_bfs.cu

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
    globalbfs.hh \
    cuda/cuda_support.hh

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

CUDA_SOURCES += \
    cuda/cuda_bfs.cu

HEADERS += \
    ../b40c/graph/bfs/csr_problem_2d.cuh \
    ../b40c/graph/bfs/enactor_multi_gpu_2d.cuh \
    ../b40c/graph/bfs/enactor_multi_gpu.cuh \
    ../b40c/util/cuda_properties.cuh \
    ../b40c/util/error_utils.cuh \
    ../b40c/util/spine.cuh \
    ../b40c/util/kernel_runtime_stats.cuh \
    ../b40c/util/cta_work_progress.cuh \
    ../b40c/util/operators.cuh \
    ../b40c/graph/bfs/csr_problem.cuh \
    ../b40c/graph/bfs/enactor_base.cuh \
    ../b40c/graph/bfs/problem_type.cuh \
    ../b40c/graph/bfs/two_phase/contract_atomic/kernel.cuh \
    ../b40c/graph/bfs/two_phase/contract_atomic/kernel_policy.cuh \
    ../b40c/graph/bfs/two_phase/expand_atomic/kernel.cuh \
    ../b40c/graph/bfs/two_phase/expand_atomic/kernel_policy.cuh \
    ../b40c/graph/bfs/partition_contract/policy.cuh \
    ../b40c/graph/bfs/partition_contract/upsweep/kernel.cuh \
    ../b40c/graph/bfs/partition_contract/upsweep/kernel_policy.cuh \
    ../b40c/graph/bfs/partition_contract/downsweep/kernel.cuh \
    ../b40c/graph/bfs/partition_contract/downsweep/kernel_policy.cuh \
    ../b40c/graph/bfs/copy/kernel.cuh \
    ../b40c/graph/bfs/copy/kernel_policy.cuh \
    cuda/cuda_bfs.h

# GPU architecture
CUDA_ARCH = 20
# NVCC flags
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -m64 -O3
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
QMAKE_LFLAGS_RELEASE += -Wl,-O3
QMAKE_LFLAGS_DEBUG += -Og



unix {

 # auto-detect CUDA path

 CUDA_DIR = "`which nvcc | sed 's,/bin/nvcc$$$$,,'`"

 QMAKE_CCXXFLAGS = $$QMAKE_CXXFLAGS
 QMAKE_CCXXFLAGS +="\"`mpicxx --showme:compile`\""
 INCLUDEPATH += $$CUDA_DIR/include ..

 QMAKE_LIBDIR += $$CUDA_DIR/lib64
 #QMAKE_LIBDIR += \"$$CUDA_DIR\"/lib

 # Path to libraries
 LIBS += -lcudart -lcuda

 cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

 cuda.commands = nvcc -c -ccbin=icc -Xcompiler $$join(QMAKE_CCXXFLAGS,",") -arch=-gencode=arch=compute_$$CUDA_ARCH,code=\"sm_$$CUDA_ARCH,compute_$$CUDA_ARCH\"  $$NVCCFLAGS $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 cuda.dependcy_type = TYPE_C
 cuda.depend_command  = nvcc -M -ccbin icc -Xcompiler $$join(QMAKE_CCXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') `mpicxx --showme:compile` $$NVCCFLAGS ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'

}

cuda.input = CUDA_SOURCES

QMAKE_EXTRA_COMPILERS += cuda
