# SamsCompressor
A JPEG like compressor written in C. Currently under production.

Important functions are **readBmp** and **writeBmp** from **bmp.h**, 
and **readSams** and **writeSams** from **sams.h**.
You can compress a **BMP** file using the **compress** function in **jpeg.h**,
and decompress a **SAMS** file using the **decompress** function.

Currently, there are some artifacts in the image, however I managed to
compress a **150kb** image to **28kb** in around **0.01** seconds, with minimal memory
overhead.


### Currently, I will work on
- making the program usable through command arguments
- reducing the artifacts in the image
- improving code readability
- improving code reliability
