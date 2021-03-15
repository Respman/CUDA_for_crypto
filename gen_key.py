#! /usr/bin/env python3

from itertools import *
import os

# максимальное количество ключей в одном файле
n = 40000000 

def main():

    os.system("rm -r ./keys")
    os.system("mkdir ./keys")

    alph = "abcdefghijklmnopqrstuvwxyz"

    mas = [alph, alph, alph, alph, alph, alph]

    out = []
    comb = set(permutations(mas))

    amnt = 0
    crt = 0
    crt_name = 0
    name = './keys/file'
    txt = '.txt'
    for mas in [[list(j) for j in i] for i in comb]:
        a1, a2, a3, a4, a5, a6 = mas
        for i in product(a1, a2, a3, a4, a5, a6):
            
            if (crt == 0):
                f = open((name+str(crt_name)+txt),'w')
                crt_name += 1

            i = list(i)
            passw = "".join(i)
            f.write(passw+'\n')
            
            crt += 1
            amnt += 1
            if (crt == n):
                crt = 0
                f.close()

    print(f"amount of keys: {amnt}")


if __name__ == '__main__':
    main()
