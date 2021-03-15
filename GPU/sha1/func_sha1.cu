/* источник: https://www.packetizer.com/security/sha1/ */


/* 
 *  Эта структура будет содержать контекстнуб информацию 
 *  для орепации хэширования
 */
typedef struct SHA1Context
{
    unsigned Message_Digest[5]; /* подписть сообщения (выходная)    */

    unsigned Length_Low;        /* длина сообщения в битах          */
    unsigned Length_High;       /* длина сообщения в битах          */

    unsigned char Message_Block[64]; /* 512-битный блок сообщения   */
    int Message_Block_Index;    /* индекс внутри блока сообщения    */

    int Computed;               /* Подпись посчитана?               */
    int Corrupted;              /* Подпись сообщения испорчена?     */
} SHA1Context;

/*
 *  Определим макрос циклического сдвига
 */
#define SHA1CircularShift(bits,word) \
                ((((word) << (bits)) & 0xFFFFFFFF) | \
                ((word) >> (32-(bits))))

/* прототипы функций */
__device__ void SHA1ProcessMessageBlock(SHA1Context *);
__device__ void SHA1PadMessage(SHA1Context *);

/*  
 *  SHA1Reset
 *
 * Эта функци инициализирует структуру SHA1Context, подготавливаясь
 * для рассчета новой подписи для сообщения.
 *
 */
__device__ void SHA1Reset(SHA1Context *context)
{
    context->Length_Low             = 0;
    context->Length_High            = 0;
    context->Message_Block_Index    = 0;

    context->Message_Digest[0]      = 0x67452301;
    context->Message_Digest[1]      = 0xEFCDAB89;
    context->Message_Digest[2]      = 0x98BADCFE;
    context->Message_Digest[3]      = 0x10325476;
    context->Message_Digest[4]      = 0xC3D2E1F0;

    context->Computed   = 0;
    context->Corrupted  = 0;
}

/*  
 *  SHA1Result
 *
 * Данная функция возвращает 160-битную поднпись в массиве 
 * Message_Digest, который находится в структуре SHA1Context.
 *
 */
 __device__ int SHA1Result(SHA1Context *context)
{

    if (context->Corrupted)
    {
        return 0;
    }

    if (!context->Computed)
    {
        SHA1PadMessage(context);
        context->Computed = 1;
    }

    return 1;
}

/*  
 *  SHA1Input
 *
 * Эта функиця принимает массив октетов как следуюшую порцию сообщения.
 *
 */
 __device__ void SHA1Input(     SHA1Context         *context,
                    char *message_array,
                    unsigned            length)
{
    if (!length)
    {
        return;
    }

    if (context->Computed || context->Corrupted)
    {
        context->Corrupted = 1;
        return;
    }

    while(length-- && !context->Corrupted)
    {
        context->Message_Block[context->Message_Block_Index++] =
                                                (*message_array & 0xFF);

        context->Length_Low += 8;
        /* обрезать до 32 бит */
        context->Length_Low &= 0xFFFFFFFF;
        if (context->Length_Low == 0)
        {
            context->Length_High++;
            /* обрезать до 32 бит */
            context->Length_High &= 0xFFFFFFFF;
            if (context->Length_High == 0)
            {
                /* сообщение слишком длинное */
                context->Corrupted = 1;
            }
        }

        if (context->Message_Block_Index == 64)
        {
            SHA1ProcessMessageBlock(context);
        }

        message_array++;
    }
}

/*  
 *  SHA1ProcessMessageBlock
 *
 * Эта функция обрабатывает следующий 512-битный кусок сообщения,
 * хранящийся в массиве Message_Block.
 * 
 */
 __device__ void SHA1ProcessMessageBlock(SHA1Context *context)
{
    const unsigned K[] =            /* Константы, определенные в схеме SHA-1   */      
    {
        0x5A827999,
        0x6ED9EBA1,
        0x8F1BBCDC,
        0xCA62C1D6
    };
    int         t;                  /* счётчик циклов               */
    unsigned    temp;               /* временное хранилице слова    */
    unsigned    W[80];              /* Последовательность слов      */
    unsigned    A, B, C, D, E;      /* Буфер слов                   */

    /*
     *  Инициализировать первые 16 слов в массиве W
     */
    for(t = 0; t < 16; t++)
    {
        W[t] = ((unsigned) context->Message_Block[t * 4]) << 24;
        W[t] |= ((unsigned) context->Message_Block[t * 4 + 1]) << 16;
        W[t] |= ((unsigned) context->Message_Block[t * 4 + 2]) << 8;
        W[t] |= ((unsigned) context->Message_Block[t * 4 + 3]);
    }

    for(t = 16; t < 80; t++)
    {
       W[t] = SHA1CircularShift(1,W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]);
    }

    A = context->Message_Digest[0];
    B = context->Message_Digest[1];
    C = context->Message_Digest[2];
    D = context->Message_Digest[3];
    E = context->Message_Digest[4];

    for(t = 0; t < 20; t++)
    {
        temp =  SHA1CircularShift(5,A) +
                ((B & C) | ((~B) & D)) + E + W[t] + K[0];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }

    for(t = 20; t < 40; t++)
    {
        temp = SHA1CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[1];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }

    for(t = 40; t < 60; t++)
    {
        temp = SHA1CircularShift(5,A) +
               ((B & C) | (B & D) | (C & D)) + E + W[t] + K[2];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }

    for(t = 60; t < 80; t++)
    {
        temp = SHA1CircularShift(5,A) + (B ^ C ^ D) + E + W[t] + K[3];
        temp &= 0xFFFFFFFF;
        E = D;
        D = C;
        C = SHA1CircularShift(30,B);
        B = A;
        A = temp;
    }

    context->Message_Digest[0] =
                        (context->Message_Digest[0] + A) & 0xFFFFFFFF;
    context->Message_Digest[1] =
                        (context->Message_Digest[1] + B) & 0xFFFFFFFF;
    context->Message_Digest[2] =
                        (context->Message_Digest[2] + C) & 0xFFFFFFFF;
    context->Message_Digest[3] =
                        (context->Message_Digest[3] + D) & 0xFFFFFFFF;
    context->Message_Digest[4] =
                        (context->Message_Digest[4] + E) & 0xFFFFFFFF;

    context->Message_Block_Index = 0;
}

/*  
 *  SHA1PadMessage
 *
 *  Согласно стандарту, сообщение должно быть дополнено кратно
 *  512 бит. Первый бит заполнения должен быть "1". Последние 64
 *  бита представляют длину исходного сообщения. Все биты в
 *  между ними должны быть равны 0. Эта функция будет дополнять сообщение
 *  в соответствии с этими правилами путем заполнения массива Message_Block
 *  соответственно. Он также вызовет SHA1ProcessMessageBlock()
 *  соответствующим образом. После выполнения данной функции по возвращаемому
 *  значению можно будет сказать, была ли испорчена подпись сообщения.
 *
 */
 __device__ void SHA1PadMessage(SHA1Context *context)
{
    /*
     * Проверим, не слишком ли мал текущий блок сообщений для хранения
     * начальных бит заполнения и длины сообщения. Если это так, мы
     * добавим блок, обрабатываем его, а затем продолжим заполнение
     * в следующий блок.
     * 
     */
    if (context->Message_Block_Index > 55)
    {
        context->Message_Block[context->Message_Block_Index++] = 0x80;
        while(context->Message_Block_Index < 64)
        {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }

        SHA1ProcessMessageBlock(context);

        while(context->Message_Block_Index < 56)
        {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }
    }
    else
    {
        context->Message_Block[context->Message_Block_Index++] = 0x80;
        while(context->Message_Block_Index < 56)
        {
            context->Message_Block[context->Message_Block_Index++] = 0;
        }
    }

    /*
     *  Храним длину сообщения в качестве последних 8-ми октетов
     */
    context->Message_Block[56] = (context->Length_High >> 24) & 0xFF;
    context->Message_Block[57] = (context->Length_High >> 16) & 0xFF;
    context->Message_Block[58] = (context->Length_High >> 8) & 0xFF;
    context->Message_Block[59] = (context->Length_High) & 0xFF;
    context->Message_Block[60] = (context->Length_Low >> 24) & 0xFF;
    context->Message_Block[61] = (context->Length_Low >> 16) & 0xFF;
    context->Message_Block[62] = (context->Length_Low >> 8) & 0xFF;
    context->Message_Block[63] = (context->Length_Low) & 0xFF;

    SHA1ProcessMessageBlock(context);
}
