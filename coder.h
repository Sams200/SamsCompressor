//
// Created by sams on 4/22/25.
//

#ifndef COMPRESS_H
#define COMPRESS_H

#include "bmp.h"
#include "sams.h"

SAMS* compress(const BMP* bmp, int quality);
BMP* decompress(const SAMS* sams);
#endif //COMPRESS_H
