 #include"func_sha1.c"
  
void sha1(long long int z, char *text, char *output, uint32_t length, long long int N, char *valid_hash, int hashlen){
  char *buf;
  int i, j;
  SHA1Context sha;
  char he[] = "0123456789ABCDEF";

  buf = (char *)malloc((hashlen+1)*sizeof(char));
  if (z < N){
    /* здесь мы высчитываем указатели на нужный пароль и считаем от него хэш */
    char *tmp_text = text + z*(length+1)*sizeof(char);
    char *tmp_output = output + z*sizeof(char);

    SHA1Reset(&sha);
    SHA1Input(&sha, tmp_text, length);
 
    /* переносим хэш из структуры в строковый массив-буфер */
    if (!SHA1Result(&sha))
    {
      tmp_output[0] = '2';
    }
    else
    {
      for(i=0; i<5; i++){
        for(j=0; j<8; j++){
          buf[i*8+7-j] = he[(sha.Message_Digest[i]&(0xf<<4*j))>>(4*j)];
        }
      }
      buf[40] = '\0';
    }
    /* сравниваем получаенный хэш с нужным нам */
     int flag = 1;
     for (i=0; i<hashlen; i++){
       if (buf[i] != valid_hash[i]){
         flag = 0;
       }
     }
     tmp_output[0] = '0'+flag;
  }
  free(buf);
}