#!/bin/bash

nvcc -std=c++11 -Xcompiler -fopenmp -DNUM_THREADS=4 main_cuda.cpp Prime*.cpp PrimeCuda.cu -o prime_cuda
