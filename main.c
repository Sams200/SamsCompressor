#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "coder.h"
#include "sams.h"
#include "bmp.h"

void compressFile(const char* src,const char* dst, int quality){
    clock_t start,end;
    start = clock();

    BMP* bmp=readBmp(src);
    SAMS* sams=compress(bmp,quality);
    writeSams(dst,sams);

    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used = %f seconds \n", cpu_time_used);

    freeBmp(bmp);
    freeSams(sams);
}

void decompressFile(const char* src, const char* dst){
    clock_t start,end;
    start = clock();

    SAMS* sams=readSams(src);
    BMP* bmp=decompress(sams);
    writeBmp(dst,bmp);

    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used = %f seconds\n", cpu_time_used);

    freeSams(sams);
    freeBmp(bmp);
}

void printUsage(const char* programName){
    printf("Usage:\n");
    printf("    %s -c|--compress <source> <destination> <quality>\n", programName);
    printf("    %s -d|--decompress <source> <destination>\n",programName);
    printf("\noptions:\n");
    printf("    -c, --compress  Compress the source BMP file\n");
    printf("    -d, --decompress  Decompress the source SAMS file\n");
    printf("    <quality>   Quality setting for compression (1-100)\n");
}
int main(int argc, char *argv[]){

    if(argc<4){
        printUsage(argv[0]);
        return 1;
    }

    bool isCompress = false;
    bool isDecompress = false;
    if(strncmp(argv[1], "-c", 2) == 0 || strncmp(argv[1], "--compress", 10) == 0){
        isCompress = true;
    }
    else if(strncmp(argv[1],"-d",2) == 0 || strncmp(argv[1], "--decompress", 10) == 0){
        isDecompress = true;
    }
    else{
        printUsage(argv[0]);
        return 1;
    }

    const char* src = argv[2];
    const char* dest = argv[3];

    if(strlen(src)==0 || strlen(dest)==0){
        printUsage(argv[0]);
        return 1;
    }

    if(isCompress){
        if(argc<5){
            printUsage(argv[0]);
            return 1;
        }

        char* endptr;
        int quality = (int)strtol(argv[4], &endptr, 10);
        if(*endptr != '\0' || quality < 1 || quality > 100){
            printUsage(argv[0]);
            return 1;
        }

        compressFile(src, dest, quality);
        return 0;
    }

    if(isDecompress){
        decompressFile(src, dest);
        return 0;
    }




    return 0;
}
