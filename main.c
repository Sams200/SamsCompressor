#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "jpeg.h"
#include "sams.h"
#include "bmp.h"


int main(int argc, char *argv[]){

    clock_t start,end;
    start = clock();

    BMP* bmp=readBmp("./files/Lena_24bits.bmp");
    SAMS* sams=compress(bmp,50);
    freeBmp(bmp);

    writeSams("./results/result.sams",sams);
    freeSams(sams);

    sams=readSams("./results/result.sams");
    bmp=decompress(sams);
    writeBmp("./results/result.bmp",bmp);

    freeSams(sams);
    freeBmp(bmp);

    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used = %f\n", cpu_time_used);
    return 0;
}
