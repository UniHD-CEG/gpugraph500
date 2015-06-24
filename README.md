This code contains a subset of Duane Merrill's BC40 repository
of GPU-related functions, including his BFS implementation used
in the paper, Scalable Graph Traversals.

All copyrights reserved to their original owners.


###Goggle c++ Test Framework

To install the package in ~/usr/gtest/ as shared libraries, together with sample build as well:

    $ mkdir ~/temp
    $ cd ~/temp
    $ wget http://googletest.googlecode.com/files/gtest-1.7.0.zip
    $ unzip gtest-1.7.0.zip 
    $ cd gtest-1.7.0
    $ mkdir mybuild
    $ cd mybuild
    $ cmake -DBUILD_SHARED_LIBS=ON -Dgtest_build_samples=ON -G"Unix Makefiles" ..
    $ make
    $ cp -r ../include/gtest ~/usr/gtest/include/
    $ cp lib*.so ~/usr/gtest/lib


To validate the installation, use the following test.cpp as a simple test example:

    	#include <gtest/gtest.h>
    	TEST(MathTest, TwoPlusTwoEqualsFour) {
    		EXPECT_EQ(2 + 2, 4);
    	}
    	
    	int main(int argc, char **argv) {
    		::testing::InitGoogleTest( &argc, argv );
    		return RUN_ALL_TESTS();
    	}
    
To compile the test:

        $ export GTEST_HOME=~/usr/gtest
        $ export LD_LIBRARY_PATH=$GTEST_HOME/lib:$LD_LIBRARY_PATH
        $ g++ -I test.cpp $GTEST_HOME/include -L $GTEST_HOME/lib -lgtest -lgtest_main -lpthread 

