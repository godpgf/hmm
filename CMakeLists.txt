cmake_minimum_required(VERSION 3.9)
project(hmm)

set(CMAKE_CXX_STANDARD 11)

add_executable(hmm test.cpp libhmm/base/matrix.h libhmm/basehmm.h libhmm/binomialhmm.h libhmm/libhmm_api.cpp libhmm/gaussianhmm.h)