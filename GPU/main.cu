#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <dirent.h> /* для рассчета количества файлов */

#include "info_about_GPU.cu"
#include "sha1/sha1.cu"

FILE* f_log;

int main(int argc, char **argv){
  
  if (argc != 2){
    printf("неправильное количество параметров\n");
  }

  int i, j;
  int fd;
	struct stat st;
	char *ptr, *d_ptr, *output, *d_output, *d_valid, *key;
  char name[128];
  int ctr;
  long long int amount_of_keys;
  long long int output_size;
  int key_len;

  /* рассчитываем количество файлов в директории "/keys/" */
  struct dirent **namelist;
  int NUMBER_OF_FILES = (scandir("../keys/", &namelist, NULL, alphasort))-2;

  printf("[*] информация о GPU:\n\n");
  int MAX_TREADS = print_info_about_GPU();
  printf("\n[*] начинаю перебор паролей\n\n");
  
/* создание файла логов */
  f_log = fopen("./log.txt", "w");

/* по порядку открываем все файлы с ключами */

  for (ctr=0; ctr<NUMBER_OF_FILES; ctr++){

/* собираем название файла (программа рассчитывает на то, что название
будет менее 128-ми символов) */
    strncpy(name, "../keys/file", 12);
    sprintf((name+12), "%d", ctr);
    strcat(name, ".txt");
    fprintf(f_log, "открываю файл: %s\n", name);

    fd = open(name,O_RDWR);
    if (fd == -1){
        fprintf(f_log, "[-] не могу открыть файл %s\n", name);
        fclose(f_log);
        return 1;
    }

/* ммапим текущий файл в оперативную память (для быстрой работы с ним) */

    fstat (fd, &st); /* рассчет размера файла */
    ptr = (char*)mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if ((long int)ptr == -1){
        fprintf(f_log, "[-] невозможен ммапинг файла  %s\n", name);
        fclose(f_log);
        return 1;
    }

    /* рассчет длины ключей для текущего файла */
    char * slash_n = strstr(ptr, "\n"); 
    if (slash_n != NULL){
      key_len = slash_n - ptr;
    } else {
      printf("[-] невозможно посчитать длину ключа\n");
      fprintf(f_log, "[-] невозможно посчитать длину ключа\n");
      /* аварийное завершение программы */
      cudaFree(d_ptr);
      cudaFree(d_output);
      cudaFree(d_valid);
      free(output);
      munmap(ptr, st.st_size);
      close(fd);
      fclose(f_log);
      return 1;
    }

    amount_of_keys = (int)(st.st_size/(key_len+1));
    output_size = amount_of_keys*sizeof(char);

    output = (char*)malloc(st.st_size);
    cudaMalloc(&d_ptr, st.st_size); 
    cudaMalloc(&d_output, output_size);
    cudaMemcpy(d_ptr, ptr, st.st_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_valid, strlen(argv[1])*sizeof(char));
    cudaMemcpy(d_valid, argv[1], strlen(argv[1])*sizeof(char), cudaMemcpyHostToDevice);

    /* Хендл event'а */
    cudaEvent_t syncEvent1, syncEvent2;
    float gpuTime;

    cudaEventCreate(&syncEvent1);    /* Создаем event1 */
    cudaEventCreate(&syncEvent2);    /* Создаем event2 */
    cudaEventRecord(syncEvent1, 0);  /* Записываем event1 */

    /* запуск SHA1 */
    long long int val = (amount_of_keys+MAX_TREADS-1)/MAX_TREADS;
    sha1<<<val, MAX_TREADS>>>(d_ptr, d_output, key_len, amount_of_keys, d_valid, strlen(argv[1]));

    cudaEventRecord(syncEvent2, 0);    /* Записываем event2 */
    cudaEventSynchronize(syncEvent2);  /* Синхронизируем event */
    cudaEventElapsedTime ( &gpuTime, syncEvent1, syncEvent2);
    fprintf(f_log, "[*] время, затраченное на рассчеты GPU: %.2f milliseconds\n", gpuTime );
    cudaEventDestroy(syncEvent1);
    cudaEventDestroy(syncEvent2);

    cudaMemcpy(output, d_output, output_size*sizeof(char), cudaMemcpyDeviceToHost);

    for (i=0; i < amount_of_keys; i++){
      key = ptr + i*(key_len+1)*sizeof(char);
      if (output[i] == '1'){
        printf("[+] подходящий ключ: ");
        for (j=0; j<key_len; j++){
          printf("%c", key[j]);
        }
        printf("\n\n");

        /* запись подходящего ключа в файл логов */
        fprintf(f_log, "[+] подходящий ключ: ");
        for (j=0; j<key_len; j++){
          fprintf(f_log, "%c", key[j]);
        }
        fprintf(f_log, "\n\n");
      } 
      if (output[i] == '2'){
        printf("[-] ошибка на ключе:\n");
        for (j=0; j<key_len; j++){
          printf("%c", key[j]);
        }
        printf("\n\n");

        /* запись ключа, на котором произошла ошибка, в файл логов */
        fprintf(f_log, "[-] ошибка на ключе:\n");
        for (j=0; j<key_len; j++){
          fprintf(f_log, "%c", key[j]);
        }
        fprintf(f_log, "\n\n");

        /* аварийное завершение программы */
        cudaFree(d_ptr);
        cudaFree(d_output);
        cudaFree(d_valid);
        free(output);
        munmap(ptr, st.st_size);
        close(fd);
        fclose(f_log);
        return 1;
      }
    }
    
    /*освобождаем занятую память */
    cudaFree(d_ptr);
    cudaFree(d_output);
    cudaFree(d_valid);
    free(output);

    munmap(ptr, st.st_size);
    close(fd);
    fprintf(f_log, "[+] закончили с файлом %s\n", name);

    if ((ctr % 2) == 0){
        printf("ctr = %d\n", ctr);
    }
  }

  fclose(f_log);
  return 0;
}