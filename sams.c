//
// Created by sams on 3/24/25.
//

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "sams.h"

#include <string.h>

SAMS* readSams(const char* fileName) {
    SAMS* sams = (SAMS*)malloc(sizeof(SAMS));
    if (sams == NULL) {
        return NULL;
    }

    sams->Y=NULL;
    sams->Cb=NULL;
    sams->Cr=NULL;

    FILE* file = fopen(fileName, "r");
    if (file == NULL) {
        free(sams);
        printf("Could not open file %s\n", fileName);
        return NULL;
    }

    // Read file header
    if(fread(&sams->file,sizeof(SAMS_FILE_HEADER),1,file) != 1) {
        fclose(file);
        free(sams);
        return NULL;
    }

    // Verify SAMS signature (0xF15A)
    if(sams->file.signature!=0xF15A) {
        fclose(file);
        free(sams);
        printf("File does not contain SAMS signature\n");
        return NULL;
    }

    // Read SAMS header
    if(fread(&sams->header,sizeof(SAMS_HEADER),1,file) != 1) {
        fclose(file);
        free(sams);
        return NULL;
    }

    const uint32_t luminanceSize=sams->header.lumLen;
    const uint32_t cbSize=sams->header.cbSize;
    const uint32_t crSize=sams->header.crSize;

    sams->Y = malloc(luminanceSize);
    sams->Cb = malloc(cbSize);
    sams->Cr = malloc(crSize);

    if(sams->Y == NULL || sams->Cb == NULL || sams->Cr == NULL) {
        if(sams->Y) free(sams->Y);
        if(sams->Cb) free(sams->Cb);
        if(sams->Cr) free(sams->Cr);
        free(sams);
        fclose(file);
        return NULL;
    }

    if (fread(sams->Y, luminanceSize, 1, file) != 1) {
        fclose(file);
        free(sams);
        return NULL;
    }
    if (fread(sams->Cb, cbSize, 1, file) != 1) {
        fclose(file);
        free(sams);
        return NULL;
    }
    if (fread(sams->Cr, crSize, 1, file) != 1) {
        fclose(file);
        free(sams);
        return NULL;
    }

    fclose(file);
    return sams;
}
int writeSams(const char* fileName, const SAMS* img) {
    if(!img || !img->Y || !img->Cb || !img->Cr) {
        return -1;
    }

    FILE* file = fopen(fileName, "wb");
    if(!file)
        return -1;

    if(fwrite(&img->file, sizeof(SAMS_FILE_HEADER),1,file) != 1) {
        fclose(file);
        return -1;
    }

    if(fwrite(&img->header, sizeof(SAMS_HEADER),1,file) != 1) {
        fclose(file);
        return -1;
    }

    if(fseek(file, img->file.offset, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }

    const uint32_t luminanceSize = img->header.lumLen;
    const uint32_t cbSize = img->header.cbSize;
    const uint32_t crSize = img->header.crSize;

    if(fwrite(img->Y, luminanceSize, 1, file) !=1) {
        fclose(file);
        return -1;
    }

    if(fwrite(img->Cb, cbSize, 1, file) !=1) {
        fclose(file);
        return -1;
    }

    if(fwrite(img->Cr, crSize, 1, file) !=1) {
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

void freeSams(SAMS* sams) {
    if(sams) {
        if (sams->Y) free(sams->Y);
        if (sams->Cb) free(sams->Cb);
        if (sams->Cr) free(sams->Cr);
        free(sams);
    }


}

SAMS* createSams(RLEPair* Y, const uint32_t lumLen, RLEPair* Cb, const uint32_t cbLen,
                    RLEPair* Cr, const uint32_t crLen, const uint32_t height, const uint32_t width,
                    const int LUMINANCE_QUANT[64], const int CHROMA_QUANT[64]){
    SAMS *sams=malloc(sizeof(SAMS));
    if(!sams) {
        perror("createSams - Could not allocate memory");
        return NULL;
    }

    sams->file.signature=0xF15A;
    sams->file.reserved=0x00000000;

    sams->header.width=width;
    sams->header.height=height;
    sams->header.lumLen=lumLen;
    sams->header.cbSize=cbLen;
    sams->header.crSize=crLen;

    sams->Y=Y;
    sams->Cb=Cb;
    sams->Cr=Cr;

    sams->file.size=sizeof(SAMS_FILE_HEADER)+sizeof(SAMS_HEADER)+sams->header.lumLen
                    +sams->header.cbSize+sams->header.crSize;

    sams->file.offset=sizeof(SAMS_FILE_HEADER)+sizeof(SAMS_HEADER);

    memcpy(sams->header.LUMINANCE_QUANT, LUMINANCE_QUANT, 64*sizeof(int32_t));
    memcpy(sams->header.CHROMA_QUANT, CHROMA_QUANT, 64*sizeof(int32_t));

    return sams;
}