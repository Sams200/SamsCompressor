# SamsCompressor  
A JPEG like compressor written in C. Not quite as fast as OpenCV  
  
**Usage:**  
&emsp;-c|--compress \<source> \<destination> \<quality>  
&emsp;-d|--decompress \<source> \<destination>  
**Options:**  
&emsp;-c, --compress &emsp;Compress the source BMP file  
&emsp;-d, --decompress &emsp;Decompress the source SAMS file  
&emsp;\<quality> &emsp;Quality setting for compression (1-100)  
    
Important functions are **readBmp** and **writeBmp** from **bmp.h**, 
and **readSams** and **writeSams** from **sams.h**.
You can compress a **BMP** file using the **compress** function in **jpeg.h**,
and decompress a **SAMS** file using the **decompress** function.

Compression performance is slightly behind OpenCV. I managed
to obtain a 7 times compression ratio, while OpenCV managed
a 9 times compression ratio.

### Currently, I will work on
- improving code readability
- improving code reliability

