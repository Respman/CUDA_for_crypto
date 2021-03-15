#! /usr/bin/env bash

rm ./a.out
gcc ./main.c -fopenmp
time ./a.out 85F734A549BCA139E4ECEF71AFE94C5C58AAD03B # хэш от ключа djnvqb