objects = memorytest.o

all: $(objects)
	/usr/local/cuda-7.0/bin/nvcc -ccbin /usr/bin/g++-4.8 -Xcompiler -m64,-pipe,-O3,-std=c++11,"`mpicxx -compile_info --showme:compile | sed -e 's,-compile_info,,' -e 's,--showme:compile,,' -e 's,g++,,' -e 's,icpc,,' -e 's,-pthread,,'`" -std=c++11 --compiler-options -fno-strict-aliasing -use_fast_math -m64 -O3 -I ""`which nvcc | sed 's,/bin/nvcc$$,,'`"/include" -I ".."  $(objects) -o app

%.o: %.cpp
	/usr/local/cuda-7.0/bin/nvcc -x cu -arch=sm_20 -ccbin /usr/bin/g++-4.8 -Xcompiler -m64,-pipe,-O3,-std=c++11,"`mpicxx -compile_info --showme:compile | sed -e 's,-compile_info,,' -e 's,--showme:compile,,' -e 's,g++,,' -e 's,icpc,,' -e 's,-pthread,,'`" -std=c++11 --compiler-options -fno-strict-aliasing -use_fast_math -m64 -O3 -I ""`which nvcc | sed 's,/bin/nvcc$$,,'`"/include" -I. -dc $< -o $@

clean:
	rm -f *.o app
