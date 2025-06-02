//
// Created by sams on 4/22/25.
//

#include "coder.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>

#include "bmp.h"
#include "sams.h"

#pragma pack(1)
#define uchar unsigned char
#define ONE_OVER_SQRT_TWO 0.70710678f

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
void downsampleChroma(void **channel, int height, int width) {
    const int newHeight = (height + 1) / 2; // Ceiling division for odd dimensions
    const int newWidth = (width + 1) / 2;
    const int newSize = newHeight * newWidth;

    void *newCh = malloc(sizeof(float) * newSize);

    const float* chp=(float*)*channel;
    float* newChp=(float*)newCh;

    // Average blocks of 2x2 pixels
    for (int i = 0; i < height; i += 2) {
        for (int j = 0; j < width; j += 2) {
            const int dy[]={0,1,0,1};
            const int dx[]={0,0,1,1};

            float sum=0;
            float amount=0;
            for (int k=0; k<4; k++) {
                const int ky=i+dy[k];
                const int kx=j+dx[k];
                const int index=ky*width+kx;

                if(isInside(height,width,ky,kx)) {
                    sum+=chp[index];
                    amount++;
                }
            }

            const int newY=i/2;
            const int newX=j/2;
            const int index=newY*newWidth + newX;
            if(amount==0) {
                newChp[index] = 128;
            }
            else {
                newChp[index] = sum/amount;
            }
        }
    }
    free(*channel);
    *channel = newChp;
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
            DCT[i*8+j] = cosf((float)(2*j+1) * (float)i * (float)M_PI/16.0f);
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

    // Shift pixels left by 128
    float temp[64];
    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            const int index=(i+y)*width+x+j;
            temp[i*8+j]=originalFloat[index]-128.0f;
        }
    }

    // Apply dct on rows
    float rowDCT[64];
    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            float s=0.0f;
            const float scale = (j==0)? ONE_OVER_SQRT_TWO:1.0f;

            for(int k=0;k<8;k++) {
                s+=(float)temp[i*8+k] * dct[j*8+k];
            }

            rowDCT[i*8+j]=s*scale;
        }
    }

    // Apply dct on columns
    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            float s=0.0f;
            const float scale = (i==0)? ONE_OVER_SQRT_TWO:1.0f;

            for(int k=0;k<8;k++) {
                s+=rowDCT[k*8+j] * dct[i*8+k];
            }

            const int index=(i+y)*width + x+j;
            originalFloat[index]=s*scale*0.25f; // 1/sqrt(2*8)=0.25
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
    17,  18,  24,  47,  70,  80,  90,  95,
    18,  21,  26,  66,  74,  84,  93,  95,
    24,  26,  56,  78,  85,  93,  95,  98,
    47,  66,  80,  86,  92,  95,  98,  99,
    70,  74,  85,  92,  95,  97,  99,  99,
    80,  84,  93,  95,  97,  99,  99,  99,
    90,  93,  95,  98,  99,  99,  99,  99,
    95,  95,  98,  99,  99,  99,  99,  99
};

/**
 * so we just divide by the quantization tables
 * a higher quality means we make the quantization table
 * smaller, and a lower quality means we make it bigger
 */
int* computeQuantTable(const int quantTable[64],unsigned int quality) {
    int* table=(int*)malloc(sizeof(int)*64);

    if(quality<1)
        quality=1;
    if(quality>100)
        quality=100;

    quality=(unsigned int)((float)quality*0.85f);

    float scaleFactor;
    if(quality<50){
        // more aggressive below 50
        scaleFactor=50.0f/(float)quality;
    }
    else{
        // smoother curve above 50
        scaleFactor= 2.0f - ((float)quality*2.0f/100.0f);
    }

    for(int i=0;i<64;i++){
        int val = (int)((float)quantTable[i] * scaleFactor + 0.5f);

        if(val<1)
            val=1;
        if(val>255)
            val=255;

        table[i]=val;
    }

    return table;
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
uint32_t encodeBlock(const void* channel, const int width, const int y, const int x, RLEPair* output, int32_t* previousDC) {
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


    // encode the DC coefficient storing the high part in zeros and the low part in value
    const int16_t dc=(int16_t)(zigZagged[0] - *previousDC);
    *previousDC=zigZagged[0];

    const uint8_t lowBits = (uint8_t)(dc & 0xFF); // low part
    const int8_t highBits = (int8_t)((dc >> 8) & 0xFF); //high part
    output[0].zeros=lowBits;
    output[0].value=highBits;

    // rle encode
    uint32_t pairCount=1;
    uint8_t zeros=0;

    for(int i=1;i<64;i++) {
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
int encodeChannel(const void* channel, const int height, const int width, RLEPair** output) {
    int totalPairs=0;
    const uint32_t maxPairs= height*width;

    RLEPair* buf=malloc(sizeof(RLEPair) * maxPairs);
    if(!buf) {
        perror("encodeChannel - Could not allocate memory");
        exit(1);
    }

    int32_t previousDC=0;
    for(int i=0;i<height;i+=8) {
        for(int j=0;j<width;j+=8) {
            const int blockPairs=(int)encodeBlock(channel,width,i,j,&buf[totalPairs],&previousDC);
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

    return (int)(totalPairs*sizeof(RLEPair));
}

void processBlockEncode(const void* channel, const int width, const int y, const int x, const int quantTable[64], const float dct[64]){
    applyDCT8x8(channel,width,y,x,dct);
    quantize(channel,width,y,x,quantTable);
}

typedef enum {
    LUMINANCE,
    CHROMA
} channel_type;

typedef struct{
    void** channel_p;
    channel_type type;
    int width;
    int height;
    int quality;
}encodeChannelArgs;

typedef struct{
    int len;
    RLEPair* rle;
    int* quantTable;
}encodeChannelRet;
void* encodeChannelFull(void* arg){
    encodeChannelArgs* args=arg;

    int width=args->width;
    int height=args->height;
    if(args->type==CHROMA){
        // downsample chroma channels
        downsampleChroma(args->channel_p,height,width);
        width=(width+1)/2;
        height=(height+1)/2;

        // make blocks with neutral strategy
        divideTo8x8(args->channel_p,height,width,PADDING_NEUTRAL);
    }
    else{
        // make blocks with replicate strategy
        divideTo8x8(args->channel_p,height,width,PADDING_REPLICATE);
    }

    height=((height+7)/8)*8;
    width=((width+7)/8)*8;

    // apply dct and quantization
    float *dct = computeDCT();
    int* quantTable;
    if(args->type==CHROMA){
        quantTable=computeQuantTable(CHROMA_QUANT,args->quality);
    }
    else{
        quantTable=computeQuantTable(LUMINANCE_QUANT,args->quality);
    }

    for(int i=0;i<height;i+=8){
        for(int j=0;j<width;j+=8){
            processBlockEncode(*(args->channel_p),width,i,j,quantTable,dct);
        }
    }

    free(dct); dct=NULL;

    // RLE encode
    RLEPair* rle;
    int rleLen=encodeChannel(*(args->channel_p),height,width,&rle);
    free(*(args->channel_p)); *(args->channel_p)=NULL;

    encodeChannelRet* ret=malloc(sizeof(encodeChannelRet));
    ret->len=rleLen;
    ret->rle=rle;
    ret->quantTable=quantTable;
    return ret;
}

SAMS* compress(const BMP* bmp, const int quality) {
    const int height=bmp->header.height;
    const int width=bmp->header.width;

    // Convert to YCbCr
    void *Y=NULL,*Cb=NULL,*Cr=NULL;
    bmpBGRtoYCbCr(bmp,&Y,&Cb,&Cr);

    pthread_t thread_y,thread_cb,thread_cr;

    // initialize arguments
    encodeChannelArgs* yArgs=malloc(sizeof(encodeChannelArgs));
    yArgs->channel_p=&Y;
    yArgs->type=LUMINANCE;
    yArgs->width=width;
    yArgs->height=height;
    yArgs->quality=quality;

    encodeChannelArgs* cbArgs=malloc(sizeof(encodeChannelArgs));
    cbArgs->channel_p=&Cb;
    cbArgs->type=CHROMA;
    cbArgs->width=width;
    cbArgs->height=height;
    cbArgs->quality=quality;

    encodeChannelArgs* crArgs=malloc(sizeof(encodeChannelArgs));
    crArgs->channel_p=&Cr;
    crArgs->type=CHROMA;
    crArgs->width=width;
    crArgs->height=height;
    crArgs->quality=quality;

    // create threads
    encodeChannelRet* yRet=NULL,*cbRet=NULL,*crRet=NULL;

    pthread_create(&thread_y,NULL,encodeChannelFull,(void*)yArgs);
    pthread_create(&thread_cb,NULL,encodeChannelFull,(void*)cbArgs);
    pthread_create(&thread_cr,NULL,encodeChannelFull,(void*)crArgs);

    pthread_join(thread_y,(void**)&yRet);
    pthread_join(thread_cb,(void**)&cbRet);
    pthread_join(thread_cr,(void**)&crRet);


    // write results
    RLEPair *YRle, *CbRle, *CrRle;
    uint32_t lumLen,cbLen,crLen;

    YRle=yRet->rle; lumLen=yRet->len;
    CbRle=cbRet->rle; cbLen=cbRet->len;
    CrRle=crRet->rle; crLen=crRet->len;

    free(yArgs);
    free(cbArgs);
    free(crArgs);

    SAMS* sams=createSams(YRle,lumLen,CbRle,cbLen,CrRle,crLen,height,width,yRet->quantTable,cbRet->quantTable);

    free(yRet->quantTable); free(yRet);
    free(cbRet->quantTable); free(cbRet);
    free(crRet->quantTable); free(crRet);

    return sams;
}

void* decodeChannel(const RLEPair* encoded, const uint32_t len, const int height, const int width) {
    const uint32_t paddedHeight=((height+7)/8)*8;
    const uint32_t paddedWidth=((width+7)/8)*8;

    int32_t *decoded=calloc(paddedHeight*paddedWidth,sizeof(int32_t));
    if(!decoded) {
        perror("decodeChannel - cannot allocate memory");
        exit(1);
    }

    uint32_t pairIndex=0;
    const uint32_t pairCount=len/sizeof(RLEPair);
    int32_t previousDC=0;

    for(int i=0;i<paddedHeight;i+=8) {
        for(int j=0;j<paddedWidth;j+=8) {
            int32_t block[64]={0};

            // handle DC coefficient
            const uint8_t lowBits = encoded[pairIndex].zeros;
            const int8_t highBits = encoded[pairIndex].value;

            int32_t dc =(int16_t)(highBits<<8) | (uint16_t)lowBits;
            dc+=previousDC;
            previousDC=dc;

            block[0]=dc;
            pairIndex++;

            int blockIndex=1;

            // read the block
            while(blockIndex<64 && pairIndex<pairCount) {
                // skip zeros
                blockIndex+=encoded[pairIndex].zeros;

                // read value
                if(blockIndex<64) {
                    block[blockIndex]= (int32_t)encoded[pairIndex].value;
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
    int32_t *originalInt=(int32_t*)channel;

    for(int i=0;i<8;i++) {
        for(int j=0;j<8;j++) {
            const int index=(i+y)*width+x+j;
            const int32_t value=originalInt[index];
            originalFloat[index]=(float)value * (float)quantTable[i*8+j];

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
                const float scale = (k==0)? ONE_OVER_SQRT_TWO:1.0f;
                s+= coeffs[k*8+j] * scale * dct[k*8+i];
            }

            colIDCT[i*8+j]=s;
        }
    }

    // rows
    for(int i=0;i<8;i++){
        for(int j=0;j<8;j++){
            float s=0.0f;

            for(int k=0;k<8;k++){
                const float scale = (k==0)? ONE_OVER_SQRT_TWO:1.0f;
                s+= colIDCT[i*8+k] * scale * dct[k*8+j];
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

void processBlockDecode(void* channel, const int width, const int y, const int x, const int quantTable[64], const float dct[64]){
    reverseQuantize(channel,width,y,x,quantTable);
    reverseDCT(channel,width,y,x,dct);
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

typedef struct{
    RLEPair *rle;
    int len;

    int height;
    int width;

    const int32_t* quantTable;

    channel_type type;
}decodeChannelArgs;
void* decodeChannelFull(void* arg){
    decodeChannelArgs *args=(decodeChannelArgs*)arg;

    int height=args->height, imgHeight=args->height;
    int width=args->width, imgWidth=args->width;

    if(args->type==CHROMA){
        height=(height+1)/2;
        width=(width+1)/2;
    }

    // decode RLE
    void* channel=decodeChannel(args->rle, args->len, height, width);
    height=((height+7)/8)*8;
    width=((width+7)/8)*8;

    // reverse quantize and dct
    float *dct = computeDCT();
    for(int i=0;i<height;i+=8) {
        for(int j=0;j<width;j+=8) {
            processBlockDecode(channel,width,i,j,args->quantTable,dct);
        }
    }
    free(dct); dct=NULL;

    // restructure without padding
    if(args->type==LUMINANCE){
        restructure(&channel,width,height,imgWidth,imgHeight);
    }
    else{
        restructure(&channel,width,height,(imgWidth+1)/2,(imgHeight+1)/2);
        height=(imgHeight+1)/2;
        width=(imgWidth+1)/2;

        // upscale chroma
        upscale(&channel,width,height,imgWidth,imgHeight);
    }

    return channel;
}
BMP* decompress(const SAMS* sams) {
    RLEPair *YRle=sams->Y;
    RLEPair *CbRle=sams->Cb;
    RLEPair *CrRle=sams->Cr;

    const int height=(int)sams->header.height;
    const int width=(int)sams->header.width;

    const int lumLen=(int)sams->header.lumLen;
    const int cbLen=(int)sams->header.cbSize;
    const int crLen=(int)sams->header.crSize;

    const int32_t *LUMINANCE_QUANT_READ = sams->header.LUMINANCE_QUANT;
    const int32_t *CHROMA_QUANT_READ = sams->header.CHROMA_QUANT;

    pthread_t thread_y,thread_cb,thread_cr;

    // initialize arguments
    decodeChannelArgs* yArgs=malloc(sizeof(decodeChannelArgs));
    yArgs->rle=YRle;
    yArgs->len=lumLen;
    yArgs->type=LUMINANCE;
    yArgs->quantTable=LUMINANCE_QUANT_READ;
    yArgs->width=width;
    yArgs->height=height;

    decodeChannelArgs* cbArgs=malloc(sizeof(decodeChannelArgs));
    cbArgs->rle=CbRle;
    cbArgs->len=cbLen;
    cbArgs->type=CHROMA;
    cbArgs->quantTable=CHROMA_QUANT_READ;
    cbArgs->width=width;
    cbArgs->height=height;

    decodeChannelArgs* crArgs=malloc(sizeof(decodeChannelArgs));
    crArgs->rle=CrRle;
    crArgs->len=crLen;
    crArgs->type=CHROMA;
    crArgs->quantTable=CHROMA_QUANT_READ;
    crArgs->width=width;
    crArgs->height=height;

    // create threads
    void *Y=NULL,*Cb=NULL,*Cr=NULL;

    pthread_create(&thread_y,NULL,decodeChannelFull,(void*)yArgs);
    pthread_create(&thread_cb,NULL,decodeChannelFull,(void*)cbArgs);
    pthread_create(&thread_cr,NULL,decodeChannelFull,(void*)crArgs);

    pthread_join(thread_y,&Y);
    pthread_join(thread_cb,&Cb);
    pthread_join(thread_cr,&Cr);

    // write results
    free(yArgs);
    free(cbArgs);
    free(crArgs);

    // convert to BGR
    BMP* bmp=createBMP24bit(width,height);
    YCbCrToBGR(Y,Cb,Cr,bmp);

    free(Y);
    free(Cb);
    free(Cr);


    return bmp;

}