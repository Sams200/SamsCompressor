//
// Created by sams on 3/19/25.
//

#ifndef BMP_H
#define BMP_H
#include <stdint.h>
#include <stdbool.h>

#pragma pack(1)
typedef struct {
    uint16_t signature;
    uint32_t size;
    uint32_t reserved;
    uint32_t offset;
} BMP_FILE_HEADER;

typedef struct {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bits;
    uint32_t compression;
    uint32_t sizeImage;
    int32_t xPelsPerMeter;
    int32_t yPelsPerMeter;
    uint32_t clrUsed;
    uint32_t clrImportant;
} BMP_HEADER;

typedef struct {
    BMP_FILE_HEADER file;
    BMP_HEADER header;
    unsigned char* data;
    uint64_t rowSize;

    bool hasTable;
    unsigned int tableSize;
    unsigned char* table;
} BMP;

BMP* readBmp(const char* fileName);

unsigned char* bmpAt(const BMP* img,const unsigned int y, const unsigned int x);

int writeBmp(const char* fileName, const BMP* bmp);

void freeBmp(BMP* bmp);

BMP* createBMP24bit(const int width, const int height);
#endif //BMP_H
