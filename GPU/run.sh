#! /usr/bin/env bash

rm ./a.out
nvcc ./main.cu
time ./a.out 2446F0A91A1E6F9995E64D782D51709075B82F54 # хэш от ключа njdhsb