#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <inttypes.h>
#include <string.h>
#include <dirent.h> /* для рассчета количества файлов */
#include <omp.h>    /* для OpenMP */

#include "sha1/sha1.c"

FILE* f_log;

int main(int argc, char **argv){
  
  if (argc != 2){
    printf("неправильное количество входных параметров\n");
  }

  long long int i, j;
  int fd;
	struct stat st;
	char *ptr, *output, *key;
  char name[128];
  int ctr;
  long long int amount_of_keys;
  long long int output_size;
  int key_len;

  /* рассчитываем количество файлов в директории "/keys/" */
  struct dirent **namelist;
  int NUMBER_OF_FILES = (scandir("../keys/", &namelist, NULL, alphasort))-2;

  printf("\n[*] начинаем перебор паролей\n\n");
  
  /* создание файла логов */
  f_log = fopen("./log.txt", "w");

/* по порядку открываем все файлы с ключами */

for (ctr=0; ctr<NUMBER_OF_FILES; ctr++){

/* собираем название файла (программа рассчитывает на то, что название
будет менее 128-ми символов) */
    strncpy(name, "../keys/file", 12);
    sprintf((name+12), "%d", ctr);
    strcat(name, ".txt");
    fprintf(f_log, "открываем файл: %s\n", name);

    fd = open(name,O_RDWR);
    if (fd == -1){
        printf("[-] не можем открыть %s\n", name);
        fprintf(f_log, "[-] не можем открыть %s\n", name);
        fclose(f_log);
        return 1;
    }

/* ммапим текущий файл в оперативную память (для быстрой работы с ним) */

    fstat (fd, &st); /* рассчет размера файла */
    ptr = (char*)mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if ((long int)ptr == -1){
      printf("[-] невозможен ммапинг файла %s\n", name);
      fprintf(f_log, "[-] невозможен ммапинг файла %s\n", name);
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
      free(output);
      munmap(ptr, st.st_size);
      close(fd);
      fclose(f_log);
      return 1;
    }

    amount_of_keys = (long long int)(st.st_size/(key_len+1));
    output_size = amount_of_keys*sizeof(char);

    output = (char*)malloc(st.st_size);

    #pragma omp parallel for
    for (i=0; i < amount_of_keys; i++){
      /* запуск SHA1 */
      sha1(i, ptr, output, key_len, amount_of_keys, argv[1], strlen(argv[1]));
    }

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
        free(output);
        munmap(ptr, st.st_size);
        close(fd);
        fclose(f_log);
        return 1;
      }
    }  
    /*освобождаем занятую память */
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