#!/bin/sh
c++ -std=c++11 -flto -mavx -mno-avx2 -pthread -g -Ofast -o overhill2 overhill2.cpp

