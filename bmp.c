//
// Created by sams on 3/19/25.
//

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "bmp.h"

#include <string.h>

#include "sams.h"

#pragma pack(1)

BMP* readBmp(const char* fileName) {
    BMP* bmp = malloc(sizeof(BMP));
    if (bmp == NULL) {
        return NULL;
    }

    bmp->data = NULL;
    bmp->table = NULL;
    bmp->hasTable = false;

    FILE *file = fopen(fileName, "rb");
    if (file == NULL) {
        free(bmp);
        perror("Unable to open file");
        return NULL;
    }

    // Read file header
    if (fread(&bmp->file, sizeof(BMP_FILE_HEADER), 1, file) != 1) {
        fclose(file);
        free(bmp);
        return NULL;
    }

    // Verify BMP signature ("BM")
    if (bmp->file.signature != 0x4D42) {
        fclose(file);
        free(bmp);
        perror("Incorrect BMP signature");
        return NULL;
    }

    // Read BMP header
    if (fread(&bmp->header, sizeof(BMP_HEADER), 1, file) != 1) {
        fclose(file);
        free(bmp);
        return NULL;
    }

    const uint16_t bits = bmp->header.bits;
    const int32_t width = bmp->header.width;
    const int32_t height = abs(bmp->header.height);

    // Read color table
    if(bits<=8) {
        uint32_t numColors=bmp->header.clrUsed;
        // If clrUsed is 0, calculate based on bit depth
        if (numColors == 0)
            numColors = 1 << bits;
        bmp->hasTable=true;
        bmp->tableSize=numColors*4;

        // Allocate and read color table
        bmp->table = (unsigned char*)malloc(bmp->tableSize);
        if (bmp->table == NULL) {
            fclose(file);
            free(bmp);
            return NULL;
        }

        if (fread(bmp->table, bmp->tableSize, 1, file) != 1) {
            free(bmp->table);
            fclose(file);
            free(bmp);
            return NULL;
        }
    }
    else{
        bmp->hasTable = false;
        bmp->table = NULL;
        bmp->tableSize = 0;
    }

    // Seek to pixel data
    if (fseek(file, bmp->file.offset, SEEK_SET) != 0) {
        fclose(file);
        free(bmp);
        return NULL;
    }

    const uint64_t rowSize = ((width * bits + 31) / 32) * 4;
    bmp->rowSize = rowSize;
    const uint64_t imageSize = rowSize * height;

    // Allocate memory for pixel data
    bmp->data = (unsigned char*)malloc(imageSize);
    if (bmp->data == NULL) {
        fclose(file);
        free(bmp);
        return NULL;
    }

    if (fread(bmp->data, rowSize*height, 1, file) != 1) {
        fclose(file);
        free(bmp);
        return NULL;
    }

    fclose(file);

    return bmp;
}

unsigned char* bmpAt(const BMP* img,const unsigned int y, const unsigned int x) {
    if(!img || !img->data)
        return NULL;

    const uint32_t height=abs(img->header.height);
    const uint32_t width=abs(img->header.width);

    if (y >= height || x >= width) {
        return NULL;
    }

    const unsigned char bits = img->header.bits;
    const unsigned int rowSize=img->rowSize;
    const unsigned int row=(height - 1 - y);

    unsigned int bytesPerPixel = bits / 8;
    if(!img->hasTable)
        return &img->data[row * img->rowSize + x * bytesPerPixel];

    if(!img->table)
        return NULL;

    if(img->header.bits==8) {
        const unsigned char index = *(img->data + row*rowSize + x);
        return &img->table[index * 4];
    }

    // Less than 8 bits per pixel
    const uint32_t byteIndex = row*rowSize + (x*bits)/8;
    const uint8_t byte = img->data[byteIndex];
    const uint8_t bitOffset = 8 - bits - ((x*bits) % 8);

    uint8_t mask;
    switch (bits) {
        case 4:
            mask = 0x0F;
        break;
        case 2:
            mask = 0x03;
        break;
        case 1:
            mask = 0x01;
        break;
        default:
            return NULL;
    }

    const uint8_t tableIndex = (byte >> bitOffset) & mask;
    return &img->table[tableIndex * 4];

}

int writeBmp(const char* fileName, const BMP* bmp){
    if(!bmp || !bmp->data)
        return -1;

    FILE* file = fopen(fileName, "wb");
    if (!file) {
        return -1;
    }

    if (fwrite(&bmp->file, sizeof(BMP_FILE_HEADER), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    if (fwrite(&bmp->header, sizeof(BMP_HEADER), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    // Write color table if present
    if (bmp->hasTable && bmp->table) {
        if (fwrite(bmp->table, bmp->tableSize, 1, file) != 1) {
            fclose(file);
            return -1;
        }
    }

    // Seek to pixel data position
    if (fseek(file, bmp->file.offset, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }

    const uint32_t height = abs(bmp->header.height);
    const uint64_t imageSize = bmp->rowSize * height;

    if(fwrite(bmp->data, imageSize, 1, file) !=1) {
        fclose(file);
        return -1;
    }


    fclose(file);
    return 0;
}

void freeBmp(BMP* bmp) {
    if (bmp) {
        if (bmp->data) free(bmp->data);
        if (bmp->hasTable && bmp->table) free(bmp->table);
        free(bmp);
    }
}

BMP* createBMP24bit(const int width, const int height) {
    BMP *bmp=malloc(sizeof(BMP));
    if(!bmp) {
        perror("createBMP - Could not allocate memory");
        return NULL;
    }

    bmp->file.signature=0x4D42;
    bmp->file.reserved=0x00000000;
    bmp->file.offset=0x36;

    const uint64_t rowSize = ((width * 24 + 31) / 32) * 4;
    bmp->rowSize=rowSize;
    bmp->file.size = rowSize*height + sizeof(BMP_HEADER) + sizeof(BMP_FILE_HEADER);

    bmp->header.size=0x28;
    bmp->header.width=width;
    bmp->header.height=height;
    bmp->header.planes=1;
    bmp->header.bits=24;
    bmp->header.compression=0;
    bmp->header.sizeImage=rowSize*height;
    bmp->header.yPelsPerMeter=2835;
    bmp->header.xPelsPerMeter=2835;
    bmp->header.clrImportant=0;
    bmp->header.clrUsed=0;

    bmp->hasTable = false;
    bmp->tableSize = 0;
    bmp->table = NULL;

    bmp->data=calloc(bmp->rowSize * height,1);
    if(!bmp->data) {
        perror("createBMP24bit - cannot allocate memory");
        return NULL;
    }

    return bmp;
}