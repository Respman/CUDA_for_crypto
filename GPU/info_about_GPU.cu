/* источник https://gist.github.com/stevendborrelli/4286842 */
/* источник информации о сетке и о потоках внутри неё:
https://www.youtube.com/watch?v=kzXjRFL-gjo */

#pragma once
#include <stdio.h>

int print_info_about_GPU() {
    int deviceCount; 
    cudaDeviceProp deviceProp;

    cudaGetDeviceCount(&deviceCount); 
    printf("Количество CUDA девайсов %d.\n", deviceCount); 

    for (int dev = 0; dev < deviceCount; dev++) {

        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                printf("CUDA GPU-девайсы не обнаружены\n");
                return -1;
            } else if (deviceCount == 1) {
                printf("Обнаружен один девайс с поддержкой CUDA\n");
            } else {
                printf("Обнаружено %d устройств, поддерживающих CUDA\n", deviceCount);
            }
        }

        printf("Для девайса #%d\n", dev); 
        printf("Название девайса:                             %s\n", deviceProp.name); 
        printf("Общее количество памяти:                      %ld\n", deviceProp.totalGlobalMem);
        printf("Обзее количество разделяеммой памяти на блок: %ld\n", 
                                                                deviceProp.sharedMemPerBlock); 
        printf("Количество статической памяти:                %ld\n", deviceProp.totalConstMem); 
        printf("Размер варпа:                                 %d\n", deviceProp.warpSize); 
        printf("Максимальное количество потоков на блок:      %d\n", deviceProp.maxThreadsDim[0]);                                          
        printf("Максимальное количество блоков в сетке:       %d\n", deviceProp.maxGridSize[0]);
        printf("Количество мультипроцессоров:                 %d\n",
                                                                deviceProp.multiProcessorCount); 
    } 
    return deviceProp.maxThreadsDim[0];
}
