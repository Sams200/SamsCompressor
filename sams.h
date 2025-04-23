//
// Created by sams on 3/24/25.
//

#ifndef SAMS_H
#define SAMS_H
#include <stdint.h>

#pragma pack(1)
typedef struct {
    uint16_t signature;
    uint32_t size;
    uint32_t reserved;
    uint32_t offset;
} SAMS_FILE_HEADER;

typedef struct {
    uint32_t width;
    uint32_t height;

    uint32_t lumLen; //size of channel in bytes
    uint32_t cbSize;
    uint32_t crSize;

    int32_t LUMINANCE_QUANT[64];
    int32_t CHROMA_QUANT[64];
} SAMS_HEADER;

typedef struct {
    SAMS_FILE_HEADER file;
    SAMS_HEADER header;

    void* Y;
    void* Cb;
    void* Cr;
} SAMS;

typedef struct {
    uint8_t zeros; // number of preceding zeros
    int8_t value; //value that follows
} RLEPair;

SAMS* readSams(const char* fileName);
int writeSams(const char* fileName, const SAMS* img);
void freeSams(SAMS* sams);
SAMS* createSams(RLEPair* Y, const uint32_t lumLen, RLEPair* Cb, const uint32_t cbLen,
                RLEPair* Cr, const uint32_t crLen, const uint32_t height, const uint32_t width,
                const int LUMINANCE_QUANT[64], const int CHROMA_QUANT[64]);

#endif //SAMS_H
