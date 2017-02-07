g++ -O3 -lm -Wall cpp_scatter.cpp -o cputest
nvcc -use_fast_math -arch=sm_20 --ptxas-options=-v -Xcompiler=-Wall cpp_scatter.cu -o gputest
