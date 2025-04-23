//
// Created by sams on 4/22/25.
//

#include "jpeg.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "bmp.h"
#include <stdbool.h>
#include <string.h>

#include <time.h>

#include "sams.h"

#pragma pack(1)
#define uchar unsigned char
#define ONE_OVER_SQRT_TWO 0.70710678f

/**
 *  So from what I understand, these are just some magic numbers
 *  we use to transform the image, since the human eye is more
 *  or less sensitive to certain colors and luminances
 *  I found these formulas on
 *  https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
 *  at paragraphs 2.5.1 and 2.5.2
 */
void bmpBGRtoYCbCr(const BMP* bmp, void** Y, void** Cb, void** Cr) {
    if(!bmp || !bmp->data || !Y || !Cb || !Cr) {
        perror("bmpBGRtoYCbCr - Invalid pointer");
        exit(1);
    }

    const uint32_t height=bmp->header.height;
    const uint32_t width=bmp->header.width;
    const uint32_t size = height * width;

    *Y=malloc(4*size);
    *Cb=malloc(4*size);
    *Cr=malloc(4*size);
    if(!*Y || !*Cb || !*Cr) {
        if(*Y) free(*Y);
        if(*Cb) free(*Cb);
        if(*Cr) free(*Cr);
        perror("bmpBGRtoYCbCr - Could not allocate memory");
        exit(1);
    }

    float* Yp=(float*)*Y;
    float* Cbp=(float*)*Cb;
    float* Crp=(float*)*Cr;

    for(int i=0; i<height;i++) {
        for(int j=0; j<width;j++) {
            const uchar* pixel = bmpAt(bmp, i, j);
            const float B = pixel[0];
            const float G = pixel[1];
            const float R = pixel[2];

            const uint32_t index = i * width + j;
            Yp[index] = (float)(0.299*R + 0.587*G + 0.114*B);
            Cbp[index] = (float)(-0.169*R - 0.331*G + 0.500*B + 128);
            Crp[index] = (float)(0.500*R - 0.419*G - 0.081*B + 128);
        }
    }

}

bool isInside(const int height, const int width, const int y, const int x) {
    if(y < 0 || y >= height || x < 0 || x >= width) return false;
    return true;
}


/**
 * Downsample in blocks of 2x2
 */
void downsampleCbCr(void **Cb, void **Cr, int height, int width) {
    const int newHeight = (height + 1) / 2; // Ceiling division for odd dimensions
    const int newWidth = (width + 1) / 2;
    const int newSize = newHeight * newWidth;

    void *newCb = malloc(sizeof(float) * newSize);
    void *newCr = malloc(sizeof(float) * newSize);

    if (!newCb || !newCr) {
        if (newCb) free(newCb);
        if (newCr) free(newCr);
        perror("downsampleCbCr - Could not allocate memory");
        exit(1);
    }

    const float* Cbp=(float*)*Cb;
    const float* Crp=(float*)*Cr;
    float* newCbp=(float*)newCb;
    float* newCrp=(float*)newCr;

    // Average blocks of 2x2 pixels
    for (int i = 0; i < height; i += 2) {
        for (int j = 0; j < width; j += 2) {
            const int dy[]={0,1,0,1};
            const int dx[]={0,0,1,1};

            float sumB=0,sumR=0;
            float amount=0;
            for (int k=0; k<4; k++) {
                const int ky=i+dy[k];
                const int kx=j+dx[k];
                const int index=ky*width+kx;

                if(isInside(height,width,ky,kx)) {
                    sumB+=Cbp[index];
                    sumR+=Crp[index];
                    amount++;
                }
            }

            const int newY=i/2;
            const int newX=j/2;
            const int index=newY*newWidth + newX;
            if(amount==0) {
                newCbp[index] = 128;
                newCrp[index] = 128;
            }
            else {
                newCbp[index] = sumB/amount;
                newCrp[index] = sumR/amount;
            }
        }
    }
    free(*Cr);
    free(*Cb);
    *Cb = newCb;
    *Cr = newCr;
}

typedef enum {
    PADDING_REPLICATE,
    PADDING_NEUTRAL
} PaddingStrategy;



/**
 *
 * divide to 8x8 blocks. User neutral for chroma and replicate for luminance
 * from now on we only work on 8x8 blocks
 */
void divideTo8x8(void** channel, const int height, const int width, const PaddingStrategy strategy) {

    // Round up to next multiple of 8
    const int paddedHeight = ((height + 7) / 8)*8;
    const int paddedWidth = ((width + 7) / 8)*8;

    float* original=(float*)*channel;
    float* padded=(float*)malloc(sizeof(float) * paddedHeight * paddedWidth);

    if(!padded) {
        perror("divideTo8x8 - Could not allocate memory");
        exit(1);
    }

    for(int i=0;i<height;i++) {
        // copy the line as is
        memcpy(&padded[i*paddedWidth],&original[i*width],width * sizeof(float));

        // fill the rest with padding
        if(width<paddedWidth) {
            float padValue;
            if(strategy==PADDING_REPLICATE) {
                padValue=original[i*width+width-1];
            }
            else {
                padValue=128;
            }

            for(int j=width;j<paddedWidth;j++) {
                padded[i*paddedWidth+j]=padValue;
            }
        }
    }

    // pad last lines
    if(height<paddedHeight) {
       for(int i=height; i<paddedHeight; i++) {
           if(strategy==PADDING_REPLICATE) {
               // copy previous row
               memcpy(&padded[i*paddedWidth],&padded[(i-1)*paddedWidth],paddedWidth * sizeof(float));
           }
           else {
               for(int j=0;j<paddedWidth;j++) {
                   padded[i*paddedWidth+j]=128;
               }
           }
       }
    }

    free(original);
    *channel=padded;
}

/**
 *
 * pre compute dct transfrom
 */
float* computeDCT() {

    float* DCT = malloc(sizeof(float) * 64);
    if (!DCT) {
        perror("computeDCT - Could not allocate memory");
        exit(1);
    }

    for(int i=0; i<8; i++) {
        for(int j=0; j<8; j++) {
            DCT[i*8+j] = cosf((float)(2*i+1) * (float)j * (float)M_PI/16.0f);
        }
    }
    return DCT;
}

/**
 * This is the worst thing I have encountered so far
 * I got this stuff from here, and I applied some optimization
 * by computing the dct on rows and collumns
 * https://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/dct.htm
 *
 * This function transforms the memory from integers to floats, which is why I used void*
 * in the first place.
 */
void applyDCT8x8(const float* channel, const int width, const int y, const int x, const float dct[64]) {

    float* originalFloat=(float*)channel;

    // Shift pixels left
    float temp[64];
    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            const int index=(i+y)*width+x+j;
            temp[i*8+j]=originalFloat[index]-128;
        }
    }

    // Apply dct on rows
    float rowDCT[64];
    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            float s=0.0f;
            for(int k=0;k<8;k++) {
                s+=(float)temp[i*8+k]*dct[k*8+j];
            }

            if(j==0)
                s*=ONE_OVER_SQRT_TWO;
            rowDCT[i*8+j]=s;
        }
    }

    // Apply dct on columns
    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            float s=0.0f;
            for(int k=0;k<8;k++) {
                s+=rowDCT[k*8+j]*dct[i*8+k];
            }

            if(i==0)
                s*=ONE_OVER_SQRT_TWO;

            const int index=(i+y)*width+x+j;
            originalFloat[index]=s*0.25f; // 1/sqrt(2*8)=0.25
        }
    }
}

/**
 * https://www.sciencedirect.com/topics/engineering/quantization-table
 */
static int LUMINANCE_QUANT[64] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};

static int CHROMA_QUANT[64] = {
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99
};

/**
 * so we just divide by the quantization tables
 * a higher quality means we make the quantization table
 * smaller, and a lower quality means we make it bigger
 */
void computeQuantTable(int quantTable[64],unsigned int quality) {
    if(quality<50)
        quality=1000/(quality+1);
    else
        quality=200-(quality*2+1);

    for(int i=0;i<64;i++) {
        const unsigned int ax=(quantTable[i]*quality+100)/100;

        if(ax<1)
            quantTable[i]=1;
        else if(ax>255)
            quantTable[i]=255;
        else
            quantTable[i]=(int)ax;
    }
}
void quantize(const void* channel, const int width, const int y, const int x, const int quantTable[64]) {

    int32_t* originalInt=(int32_t*)channel;
    float* originalFloat=(float*)channel;

    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            const int index=(i+y)*width+x+j;
            originalInt[index]=(int32_t)roundf(originalFloat[index]/(float)quantTable[i*8+j]);
        }
    }
}

/**
 *
 * 0  1  5  6  14 15 27 28
 * 2  4  7  13 16 26 29 42
 * 3  8  12 17 25 30 41 43
 * 9  11 18 24 31 40 44 53
 * 10 19 23 32 39 45 52 54
 * 20 22 33 38 46 51 55 60
 * 21 34 37 47 50 56 59 61
 * 35 36 48 49 57 58 62 63
 */

/**
 * T_T
 */
static const int zigZagTable[64] = {
    0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63
};

/**
 * ZigZag and RLE encode an 8x8 block
 * Returns the number of pairs created
 */
uint32_t encodeBlock(const void* channel, const int width, const int y, const int x, RLEPair* output) {
    int32_t* originalInt=(int32_t*)channel;

    // zigzag
    int32_t zigZagged[64];
    for(int i=0;i<64;i++) {
        // get x and y within the block
        const int blockY=zigZagTable[i]/8;
        const int blockX=zigZagTable[i]%8;

        const int index=(y+blockY)*width+x+blockX;
        zigZagged[i]=originalInt[index];
    }


    // rle encode
    uint32_t pairCount=0;
    uint8_t zeros=0;

    for(int i=0;i<64;i++) {
        const int val=zigZagged[i];

        if(val==0 && zeros<255) {
            zeros++;
            continue;
        }

        int8_t clampedVal;
        if(val>127) clampedVal=127;
        else if(val<-128) clampedVal=-128;
        else clampedVal=(int8_t)val;

        output[pairCount].zeros=zeros;
        output[pairCount].value=clampedVal;
        pairCount++;

        zeros=0;
    }

    if(zeros>0) {
        output[pairCount].zeros=zeros;
        output[pairCount].value=0;
        pairCount++;
    }

    return pairCount;
}

/**
 * Encode a channel
 * Return the size in bytes
 */
uint32_t encodeChannel(const void* channel, const int height, const int width, RLEPair** output) {
    uint32_t totalPairs=0;
    const uint32_t maxPairs= height*width;

    RLEPair* buf=malloc(sizeof(RLEPair) * maxPairs);
    if(!buf) {
        perror("encodeChannel - Could not allocate memory");
        exit(1);
    }

    for(int i=0;i<height;i+=8) {
        for(int j=0;j<width;j+=8) {
            const uint32_t blockPairs=encodeBlock(channel,width,i,j,&buf[totalPairs]);
            totalPairs+=blockPairs;
        }
    }

    RLEPair* resized=realloc(buf,sizeof(RLEPair) * totalPairs);
    if(resized) {
        *output=resized;
    }
    else {
        // ?? I dont know what to do here
        perror("encodeChannel - Could not reallocate memory");
        *output=buf;
    }

    return totalPairs*sizeof(RLEPair);
}

SAMS* compress(const BMP* bmp, unsigned int quality) {
    if(quality>100) quality=100;

    const int height=bmp->header.height;
    const int width=bmp->header.width;

    // Convert to YCbCr
    void *Y=NULL,*Cb=NULL,*Cr=NULL;
    bmpBGRtoYCbCr(bmp,&Y,&Cb,&Cr);

    // Downsample Cb and Cr
    downsampleCbCr(&Cb,&Cr,height,width);
    int chromaWidth=(width+1)/2;
    int chromaHeight=(height+1)/2;

    // Divide to 8x8 blocks
    divideTo8x8(&Y,height,width,PADDING_REPLICATE);
    divideTo8x8(&Cb,chromaHeight,chromaWidth,PADDING_NEUTRAL);
    divideTo8x8(&Cr,chromaHeight,chromaWidth,PADDING_NEUTRAL);

    // update dimensions
    const int lumHeight=((height+7)/8)*8;
    const int lumWidth=((width+7)/8)*8;
    chromaHeight=((chromaHeight+7)/8)*8;
    chromaWidth=((chromaWidth+7)/8)*8;

    // Apply DCT
    float *dct = computeDCT();
    for(int i=0;i<lumHeight;i+=8) {
        for(int j=0;j<lumWidth;j+=8) {
            applyDCT8x8(Y,lumWidth,i,j,dct);
        }
    }

    for(int i=0;i<chromaHeight;i+=8) {
        for(int j=0;j<chromaWidth;j+=8) {
            applyDCT8x8(Cb,chromaWidth,i,j,dct);
            applyDCT8x8(Cr,chromaWidth,i,j,dct);
        }
    }

    free(dct); dct=NULL;

    // Quantize blocks

    computeQuantTable(LUMINANCE_QUANT,quality);
    computeQuantTable(CHROMA_QUANT,quality);

    for(int i=0;i<lumHeight;i+=8) {
        for(int j=0;j<lumWidth;j+=8) {
            quantize(Y,lumWidth,i,j,LUMINANCE_QUANT);
        }
    }

    for(int i=0;i<chromaHeight;i+=8) {
        for(int j=0;j<chromaWidth;j+=8) {
            quantize(Cb,chromaWidth,i,j,CHROMA_QUANT);
            quantize(Cr,chromaWidth,i,j,CHROMA_QUANT);
        }
    }

    // RLE
    RLEPair *YRle, *CbRle, *CrRle;
    const uint32_t lumLen=encodeChannel(Y,lumHeight,lumWidth,&YRle);
    free(Y);
    const uint32_t cbLen=encodeChannel(Cb,chromaHeight,chromaWidth,&CbRle);
    free(Cb);
    const uint32_t crLen=encodeChannel(Cr,chromaHeight,chromaWidth,&CrRle);
    free(Cr);

    SAMS* sams=createSams(YRle,lumLen,CbRle,cbLen,CrRle,crLen,height,width,LUMINANCE_QUANT,CHROMA_QUANT);

    return sams;
}

void* decodeChannel(const RLEPair* encoded, const uint32_t len, const int height, const int width) {
    const uint32_t paddedHeight=((height+7)/8)*8;
    const uint32_t paddedWidth=((width+7)/8)*8;

    float *decoded=calloc(paddedHeight*paddedWidth,sizeof(float));
    if(!decoded) {
        perror("decodeChannel - cannot allocate memory");
        exit(1);
    }

    uint32_t pairIndex=0;
    const uint32_t pairCount=len/sizeof(RLEPair);

    for(int i=0;i<paddedHeight;i+=8) {
        for(int j=0;j<paddedWidth;j+=8) {
            float block[64]={0};
            int blockIndex=0;

            // read the block
            while(blockIndex<64 && pairIndex<pairCount) {
                // skip zeros
                blockIndex+=encoded[pairIndex].zeros;

                // read value
                if(blockIndex<64) {
                    block[blockIndex]= encoded[pairIndex].value;
                    blockIndex++;
                }
                pairIndex++;
            }

            // reverse the zigzag
            for(int k=0;k<64;k++) {
                const int y=zigZagTable[k]/8;
                const int x=zigZagTable[k]%8;

                const uint32_t index=(i+y)*paddedWidth+x+j;
                decoded[index]=block[k];
            }
        }
    }

    return decoded;
}

void reverseQuantize(void* channel, const int width, const int y, const int x, const int quantTable[64]) {
    float* originalFloat=(float*)channel;

    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            const int index=(i+y)*width+x+j;
            originalFloat[index]=originalFloat[index] * (float)quantTable[i*8+j];
        }
    }
}

void reverseDCT(void* channel, const int width, const int y, const int x, const float dct[64]) {
    float* channelFloat = (float*)channel;

    float coeffs[64];
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            const int index=(i+y)*width + x+j;
            coeffs[i*8+j]=channelFloat[index];
        }
    }

    // columns
    float colIDCT[64];
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            float s=0.0f;
            for(int k=0;k<8;k++){
                float factor=coeffs[k*8+j];
                if(k==0)
                    factor*=ONE_OVER_SQRT_TWO;
                s+=factor*dct[i*8+k];
            }
            colIDCT[i*8+j]=s;
        }
    }

    // rows
    float result[64];
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            float s=0.0f;
            for(int k=0;k<8;k++){
                float factor=colIDCT[i*8+k];
                if(k==0)
                    factor*=ONE_OVER_SQRT_TWO;
                s+=factor*dct[j*8+k];
            }

            s*=0.25f;
            s+=128.0f;

            if(s<0.0f) s=0.0f;
            if(s>255.0f) s=255.0f;

            const int index=(i+y)*width + x+j;
            channelFloat[index]=s;
        }
    }
}

void restructure(void** channel, const int paddedWidth, const int paddedHeight, const int imgWidth, const int imgHeight) {
    float *padded=(float*)*channel;
    float *original=malloc(sizeof(float) * imgWidth * imgHeight);
    if(!original) {
        perror("restructure - cannot allocate memory");
        exit(1);
    }

    for(int i=0;i<imgHeight;i++) {
        memcpy(&original[i*imgWidth],&padded[i*paddedWidth],sizeof(float) * imgWidth);
    }

    free(padded);
    *channel=original;
}

void upscale(void **channel, const int dsWidth, const int dsHeight, const int imgWidth, const int imgHeight) {
    float* originalFloat=(float*)*channel;
    float* newChannel=malloc(sizeof(float) * imgWidth * imgHeight);
    if(!newChannel) {
        perror("upscale - cannot allocate memory");
        exit(1);
    }

    const int dy[]={0, 0, 1, 1};
    const int dx[]={0, 1 ,0 ,1};

    for(int i=0;i<dsHeight;i++) {
        for(int j=0;j<dsWidth;j++) {
            const float value = originalFloat[i*dsWidth + j];

            for(int k=0;k<4;k++) {
                const int y=2*i+dy[k];
                const int x=2*j+dx[k];

                if(isInside(imgHeight,imgWidth,y,x)) {
                    newChannel[y*imgWidth+x]=value;
                }
            }
        }
    }

    free(originalFloat);
    *channel=newChannel;
}

void YCbCrToBGR(void* Y, void* CB, void* CR, const BMP* bmp) {
    const int height = bmp->header.height;
    const int width = bmp->header.width;

    float *Yp=Y;
    float *CBp=CB;
    float *CRp=CR;

    for(int i=0;i<height;i++) {
        for(int j=0;j<width;j++) {
            const int index=i*width+j;

            const float y=Yp[index];
            const float cb=CBp[index];
            const float cr=CRp[index];

            int r = (int) (y + 1.40200 * (cr - 0x80));
            int g = (int) (y - 0.34414 * (cb - 0x80) - 0.71414 * (cr - 0x80));
            int b = (int) (y + 1.77200 * (cb - 0x80));

            if(r<0) r=0;
            if(r>255) r=255;
            if(g<0) g=0;
            if(g>255) g=255;
            if(b<0) b=0;
            if(b>255) b=255;

            uchar* pixel=bmpAt(bmp,i,j);
            pixel[0]=(uchar)b;
            pixel[1]=(uchar)g;
            pixel[2]=(uchar)r;
        }
    }
}

BMP* decompress(const SAMS* sams) {
    RLEPair *YRle=sams->Y;
    RLEPair *CbRle=sams->Cb;
    RLEPair *CrRle=sams->Cr;

    const int height=(int)sams->header.height;
    const int width=(int)sams->header.width;

    int lumHeight=height;
    int lumWidth=width;
    int chromaHeight=(height+1)/2;
    int chromaWidth=(width+1)/2;

    const int lumLen=(int)sams->header.lumLen;
    const int cbLen=(int)sams->header.cbSize;
    const int crLen=(int)sams->header.crSize;

    const int32_t *LUMINANCE_QUANT_READ = sams->header.LUMINANCE_QUANT;
    const int32_t *CHROMA_QUANT_READ = sams->header.CHROMA_QUANT;

    // decode RLE
    void *Y = decodeChannel(YRle, lumLen, lumHeight, lumWidth);
    void *Cb = decodeChannel(CbRle, cbLen, chromaHeight, chromaWidth);
    void *Cr= decodeChannel(CrRle, crLen, chromaHeight, chromaWidth);

    lumHeight=((lumHeight+7)/8)*8;
    lumWidth=((lumWidth+7)/8)*8;
    chromaHeight=((chromaHeight+7)/8)*8;
    chromaWidth=((chromaWidth+7)/8)*8;

    // reverse quantize
    for(int i=0;i<lumHeight;i+=8) {
        for(int j=0;j<lumWidth;j+=8) {
            reverseQuantize(Y,lumWidth,i,j,LUMINANCE_QUANT_READ);
        }
    }

    for(int i=0;i<chromaHeight;i+=8) {
        for(int j=0;j<chromaWidth;j+=8) {
            reverseQuantize(Cb,chromaWidth,i,j,CHROMA_QUANT_READ);
            reverseQuantize(Cr,chromaWidth,i,j,CHROMA_QUANT_READ);
        }
    }

    // reverse dct
    float *dct = computeDCT();
    for(int i=0;i<lumHeight;i+=8) {
        for(int j=0;j<lumWidth;j+=8) {
            reverseDCT(Y,lumWidth,i,j,dct);
        }
    }

    for(int i=0;i<chromaHeight;i+=8) {
        for(int j=0;j<chromaWidth;j+=8) {
            reverseDCT(Cb,chromaWidth,i,j,dct);
            reverseDCT(Cr,chromaWidth,i,j,dct);
        }
    }

    free(dct); dct=NULL;

    // restructure without padding
    restructure(&Y,lumWidth,lumHeight,width,height);
    restructure(&Cb,chromaWidth,chromaHeight,(width+1)/2,(height+1)/2);
    restructure(&Cr,chromaWidth,chromaHeight,(width+1)/2,(height+1)/2);
    chromaHeight=(height+1)/2;
    chromaWidth=(width+1)/2;

    // upscale Cb and Cr
    upscale(&Cb,chromaWidth,chromaHeight,width,height);
    upscale(&Cr,chromaWidth,chromaHeight,width,height);

    // convert to BGR
    BMP* bmp=createBMP24bit(width,height);
    YCbCrToBGR(Y,Cb,Cr,bmp);

    free(Y);
    free(Cb);
    free(Cr);


    return bmp;

}