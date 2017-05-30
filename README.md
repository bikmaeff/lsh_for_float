code for building the project:
$ mkdir build 

$ cd build 

$ cmake ../

$ make

takes about 4 minutes


code for compiling matlab interface:

mex nearest_neighbors.cpp -I$FLANN_ROOT/src/cpp -L$HDF5_ROOT/1.10.1/lib -lhdf5 -L$FLANN_ROOT/build/lib -lflann


code for runing the LSH method. 

[index, search_params ] = flann_build_index(dataset, struct('algorithm','lsh', 'table_number',12, 'key_size',20, 'r', 400));

[result, ndists] = flann_search(index, testset, 6, search_params);

The library will automatically detect the datatype, if data is float type(i.e. single in matlab),  the added algorithm will be used for ANNS.


FLANN - Fast Library for Approximate Nearest Neighbors
======================================================

FLANN is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. It contains a collection of algorithms we found to work best for nearest neighbor search and a system for automatically choosing the best algorithm and optimum parameters depending on the dataset.
FLANN is written in C++ and contains bindings for the following languages: C, MATLAB, Python, and Ruby.

