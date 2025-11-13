// Grid is 16384 x 16384, each cell is one bit, each uint64_t holds 64 cells
// So the grid is represented as 256 x 16384 uint64_t elements
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

__global__ void multistepKernel(uint64_t* globalData, int height, int width, int steps, int iterations) {
    // Shared memory for 16 warps, each warp has 256 uint64_t (128 rows × 2 columns)
    __shared__ uint64_t warpStorage[16][256];


    // Thread/block coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Warp identification
    int warpId = (ty / 32) + (tx * 4);  // 16 warps per block
    int laneId = ty % 32;               // 0–31 within the warp

    // Global grid position
    int globalX = (blockIdx.x * blockDim.x) + tx;
    int globalXleft  = (globalX - 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)
    int globalXright = (globalX + 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)

    float yQuarter = (ty / 32) / 4.0f;
    int centralWarpStartY = (blockIdx.y * 2 * blockDim.y) + (256 * yQuarter);
    int centralWarpEndY = centralWarpStartY + 64;

    int haloStartY = centralWarpStartY - 32;
    int haloEndY = centralWarpEndY + 32;
    
    for (int i = 0; i < 4; i++){
        int localY = haloStartY + (laneId * 4) + i;
        int globalY = (localY) & (height - 1);
        uint64_t left = globalData[(globalY * width) + globalXleft];
        uint64_t middle = globalData[(globalY * width) + globalX];
        uint64_t right = globalData[(globalY * width) + globalXright];
        warpStorage[warpId][(laneId * 8) + (i * 2)] = (left << 32) | (middle >> 32);   
        warpStorage[warpId][(laneId * 8) + (i * 2) + 1] = (middle << 32) | (right >> 32); 
    }

    __syncthreads();

    if (laneId == 0){
        uint64_t r00 = 0ULL;
        uint64_t r01 = 0ULL;
        uint64_t r10 = warpStorage[warpId][0]
        uint64_t r11 = warpStorage[warpId][1]
        uint64_t r20 = warpStorage[warpId][2]
        uint64_t r21 = warpStorage[warpId][3]
        uint64_t r30 = warpStorage[warpId][4]
        uint64_t r31 = warpStorage[warpId][5]
        uint64_t r40 = warpStorage[warpId][6]
        uint64_t r41 = warpStorage[warpId][7]
        uint64_t r50 = warpStorage[warpId][8]
        uint64_t r51 = warpStorage[warpId][9]

        for (int i = 0; i < 32; i++){
            // Storage organized as [row][column]
            uint64_t storage[6][2];
            storage[0][0] = r00; storage[0][1] = r01;
            storage[1][0] = r10; storage[1][1] = r11;
            storage[2][0] = r20; storage[2][1] = r21;
            storage[3][0] = r30; storage[3][1] = r31;
            storage[4][0] = r40; storage[4][1] = r41;
            storage[5][0] = r50; storage[5][1] = r51;
            
            uint64_t output[4][2];  // 4 output rows, 2 columns
            
            // Reuse chains for column 0
            uint32_t c0_left_top_xor, c0_left_mid_xor, c0_left_top_maj, c0_left_mid_maj;
            uint32_t c0_right_top_xor, c0_right_mid_xor, c0_right_top_maj, c0_right_mid_maj;
            
            // Reuse chains for column 1
            uint32_t c1_left_top_xor, c1_left_mid_xor, c1_left_top_maj, c1_left_mid_maj;
            uint32_t c1_right_top_xor, c1_right_mid_xor, c1_right_top_maj, c1_right_mid_maj;
            
            // Process 4 output rows
            #pragma unroll //encourage compiler to unroll for loop and put into thraed registers check via compiling with: nvcc -Xptxas -v kernel.cu -o kernel.o, and executing it, should have line ptxas info
            for (int row = 1; row <= 4; row++) {
                
                // ========== COLUMN 0 ==========
                {
                    // Split into 32-bit halves
                    uint32_t left_top = (uint32_t)(storage[row-1][0] >> 32);
                    uint32_t right_top = (uint32_t)(storage[row-1][0]);
                    uint32_t left_mid = (uint32_t)(storage[row][0] >> 32);
                    uint32_t right_mid = (uint32_t)(storage[row][0]);
                    uint32_t left_bot = (uint32_t)(storage[row+1][0] >> 32);
                    uint32_t right_bot = (uint32_t)(storage[row+1][0]);
                    
                    // For wrapping from column 1
                    uint32_t col1_left_top = (uint32_t)(storage[row-1][1] >> 32);
                    uint32_t col1_left_mid = (uint32_t)(storage[row][1] >> 32);
                    uint32_t col1_left_bot = (uint32_t)(storage[row+1][1] >> 32);
                    
                    uint64_t output_val = 0;
                    
                    // --- LEFT HALF (upper 32 bits) ---
                    {
                        const uint32_t a0 = left_top >> 1;
                        const uint32_t a1 = left_top;
                        const uint32_t a2 = (left_top << 1) | (right_top >> 31);
                        const uint32_t a3 = left_mid >> 1;
                        const uint32_t center = left_mid;
                        const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
                        const uint32_t a5 = left_bot >> 1;
                        const uint32_t a6 = left_bot;
                        const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        // Magic stage
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_left_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_left_mid_maj), "r"(c0_left_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val = ((uint64_t)result << 32);
                        
                        c0_left_top_xor = c0_left_mid_xor;
                        c0_left_mid_xor = bottom_xor;
                        c0_left_top_maj = c0_left_mid_maj;
                        c0_left_mid_maj = bottom_maj;
                    }
                    
                    // --- RIGHT HALF (lower 32 bits) ---
                    {
                        const uint32_t a0 = (left_top << 31) | (right_top >> 1);
                        const uint32_t a1 = right_top;
                        const uint32_t a2 = (right_top << 1) | (col1_left_top >> 31);  // Wrap to column 1
                        const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
                        const uint32_t center = right_mid;
                        const uint32_t a4 = (right_mid << 1) | (col1_left_mid >> 31);
                        const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
                        const uint32_t a6 = right_bot;
                        const uint32_t a7 = (right_bot << 1) | (col1_left_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_right_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_right_mid_maj), "r"(c0_right_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val |= (uint64_t)result;
                        
                        c0_right_top_xor = c0_right_mid_xor;
                        c0_right_mid_xor = bottom_xor;
                        c0_right_top_maj = c0_right_mid_maj;
                        c0_right_mid_maj = bottom_maj;
                    }
                    
                    output[row-1][0] = output_val;
                }
                
                // ========== COLUMN 1 ==========
                {
                    uint32_t left_top = (uint32_t)(storage[row-1][1] >> 32);
                    uint32_t right_top = (uint32_t)(storage[row-1][1]);
                    uint32_t left_mid = (uint32_t)(storage[row][1] >> 32);
                    uint32_t right_mid = (uint32_t)(storage[row][1]);
                    uint32_t left_bot = (uint32_t)(storage[row+1][1] >> 32);
                    uint32_t right_bot = (uint32_t)(storage[row+1][1]);
                    
                    // For wrapping from column 0
                    uint32_t col0_right_top = (uint32_t)(storage[row-1][0]);
                    uint32_t col0_right_mid = (uint32_t)(storage[row][0]);
                    uint32_t col0_right_bot = (uint32_t)(storage[row+1][0]);
                    
                    uint64_t output_val = 0;
                    
                    // --- LEFT HALF (upper 32 bits) ---
                    {
                        const uint32_t a0 = (col0_right_top << 1) | (left_top >> 31);  // Wrap from column 0
                        const uint32_t a1 = left_top;
                        const uint32_t a2 = (left_top << 1) | (right_top >> 31);
                        const uint32_t a3 = (col0_right_mid << 1) | (left_mid >> 31);
                        const uint32_t center = left_mid;
                        const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
                        const uint32_t a5 = (col0_right_bot << 1) | (left_bot >> 31);
                        const uint32_t a6 = left_bot;
                        const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_left_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_left_mid_maj), "r"(c1_left_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val = ((uint64_t)result << 32);
                        
                        c1_left_top_xor = c1_left_mid_xor;
                        c1_left_mid_xor = bottom_xor;
                        c1_left_top_maj = c1_left_mid_maj;
                        c1_left_mid_maj = bottom_maj;
                    }
                    
                    // --- RIGHT HALF (lower 32 bits) ---
                    {
                        const uint32_t a0 = (left_top << 31) | (right_top >> 1);
                        const uint32_t a1 = right_top;
                        const uint32_t a2 = right_top << 1;  // Wraps to column 0's left half
                        const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
                        const uint32_t center = right_mid;
                        const uint32_t a4 = right_mid << 1;
                        const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
                        const uint32_t a6 = right_bot;
                        const uint32_t a7 = right_bot << 1;
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_right_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_right_mid_maj), "r"(c1_right_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val |= (uint64_t)result;
                        
                        c1_right_top_xor = c1_right_mid_xor;
                        c1_right_mid_xor = bottom_xor;
                        c1_right_top_maj = c1_right_mid_maj;
                        c1_right_mid_maj = bottom_maj;
                    }
                    
                    output[row-1][1] = output_val;
                }
            }
            //warpshiftStuff
        }

    }else if (laneId == 31){
        uint64_t r00 = warpStorage[warpId][246]
        uint64_t r01 = warpStorage[warpId][247]
        uint64_t r10 = warpStorage[warpId][248]
        uint64_t r11 = warpStorage[warpId][249]
        uint64_t r20 = warpStorage[warpId][250]
        uint64_t r21 = warpStorage[warpId][251]
        uint64_t r30 = warpStorage[warpId][252]
        uint64_t r31 = warpStorage[warpId][253]
        uint64_t r40 = warpStorage[warpId][254]
        uint64_t r41 = warpStorage[warpId][255]
        uint64_t r50 = 0ULL;
        uint64_t r51 = 0ULL;

        for (int i = 0; i < 32; i++){
            // Storage organized as [row][column]
            uint64_t storage[6][2];
            storage[0][0] = r00; storage[0][1] = r01;
            storage[1][0] = r10; storage[1][1] = r11;
            storage[2][0] = r20; storage[2][1] = r21;
            storage[3][0] = r30; storage[3][1] = r31;
            storage[4][0] = r40; storage[4][1] = r41;
            storage[5][0] = r50; storage[5][1] = r51;
            
            uint64_t output[4][2];  // 4 output rows, 2 columns
            
            // Reuse chains for column 0
            uint32_t c0_left_top_xor, c0_left_mid_xor, c0_left_top_maj, c0_left_mid_maj;
            uint32_t c0_right_top_xor, c0_right_mid_xor, c0_right_top_maj, c0_right_mid_maj;
            
            // Reuse chains for column 1
            uint32_t c1_left_top_xor, c1_left_mid_xor, c1_left_top_maj, c1_left_mid_maj;
            uint32_t c1_right_top_xor, c1_right_mid_xor, c1_right_top_maj, c1_right_mid_maj;
            
            // Process 4 output rows
            #pragma unroll //encourage compiler to unroll for loop and put into thraed registers check via compiling with: nvcc -Xptxas -v kernel.cu -o kernel.o, and executing it, should have line ptxas info
            for (int row = 1; row <= 4; row++) {
                
                // ========== COLUMN 0 ==========
                {
                    // Split into 32-bit halves
                    uint32_t left_top = (uint32_t)(storage[row-1][0] >> 32);
                    uint32_t right_top = (uint32_t)(storage[row-1][0]);
                    uint32_t left_mid = (uint32_t)(storage[row][0] >> 32);
                    uint32_t right_mid = (uint32_t)(storage[row][0]);
                    uint32_t left_bot = (uint32_t)(storage[row+1][0] >> 32);
                    uint32_t right_bot = (uint32_t)(storage[row+1][0]);
                    
                    // For wrapping from column 1
                    uint32_t col1_left_top = (uint32_t)(storage[row-1][1] >> 32);
                    uint32_t col1_left_mid = (uint32_t)(storage[row][1] >> 32);
                    uint32_t col1_left_bot = (uint32_t)(storage[row+1][1] >> 32);
                    
                    uint64_t output_val = 0;
                    
                    // --- LEFT HALF (upper 32 bits) ---
                    {
                        const uint32_t a0 = left_top >> 1;
                        const uint32_t a1 = left_top;
                        const uint32_t a2 = (left_top << 1) | (right_top >> 31);
                        const uint32_t a3 = left_mid >> 1;
                        const uint32_t center = left_mid;
                        const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
                        const uint32_t a5 = left_bot >> 1;
                        const uint32_t a6 = left_bot;
                        const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        // Magic stage
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_left_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_left_mid_maj), "r"(c0_left_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val = ((uint64_t)result << 32);
                        
                        c0_left_top_xor = c0_left_mid_xor;
                        c0_left_mid_xor = bottom_xor;
                        c0_left_top_maj = c0_left_mid_maj;
                        c0_left_mid_maj = bottom_maj;
                    }
                    
                    // --- RIGHT HALF (lower 32 bits) ---
                    {
                        const uint32_t a0 = (left_top << 31) | (right_top >> 1);
                        const uint32_t a1 = right_top;
                        const uint32_t a2 = (right_top << 1) | (col1_left_top >> 31);  // Wrap to column 1
                        const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
                        const uint32_t center = right_mid;
                        const uint32_t a4 = (right_mid << 1) | (col1_left_mid >> 31);
                        const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
                        const uint32_t a6 = right_bot;
                        const uint32_t a7 = (right_bot << 1) | (col1_left_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_right_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_right_mid_maj), "r"(c0_right_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val |= (uint64_t)result;
                        
                        c0_right_top_xor = c0_right_mid_xor;
                        c0_right_mid_xor = bottom_xor;
                        c0_right_top_maj = c0_right_mid_maj;
                        c0_right_mid_maj = bottom_maj;
                    }
                    
                    output[row-1][0] = output_val;
                }
                
                // ========== COLUMN 1 ==========
                {
                    uint32_t left_top = (uint32_t)(storage[row-1][1] >> 32);
                    uint32_t right_top = (uint32_t)(storage[row-1][1]);
                    uint32_t left_mid = (uint32_t)(storage[row][1] >> 32);
                    uint32_t right_mid = (uint32_t)(storage[row][1]);
                    uint32_t left_bot = (uint32_t)(storage[row+1][1] >> 32);
                    uint32_t right_bot = (uint32_t)(storage[row+1][1]);
                    
                    // For wrapping from column 0
                    uint32_t col0_right_top = (uint32_t)(storage[row-1][0]);
                    uint32_t col0_right_mid = (uint32_t)(storage[row][0]);
                    uint32_t col0_right_bot = (uint32_t)(storage[row+1][0]);
                    
                    uint64_t output_val = 0;
                    
                    // --- LEFT HALF (upper 32 bits) ---
                    {
                        const uint32_t a0 = (col0_right_top << 1) | (left_top >> 31);  // Wrap from column 0
                        const uint32_t a1 = left_top;
                        const uint32_t a2 = (left_top << 1) | (right_top >> 31);
                        const uint32_t a3 = (col0_right_mid << 1) | (left_mid >> 31);
                        const uint32_t center = left_mid;
                        const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
                        const uint32_t a5 = (col0_right_bot << 1) | (left_bot >> 31);
                        const uint32_t a6 = left_bot;
                        const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_left_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_left_mid_maj), "r"(c1_left_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val = ((uint64_t)result << 32);
                        
                        c1_left_top_xor = c1_left_mid_xor;
                        c1_left_mid_xor = bottom_xor;
                        c1_left_top_maj = c1_left_mid_maj;
                        c1_left_mid_maj = bottom_maj;
                    }
                    
                    // --- RIGHT HALF (lower 32 bits) ---
                    {
                        const uint32_t a0 = (left_top << 31) | (right_top >> 1);
                        const uint32_t a1 = right_top;
                        const uint32_t a2 = right_top << 1;  // Wraps to column 0's left half
                        const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
                        const uint32_t center = right_mid;
                        const uint32_t a4 = right_mid << 1;
                        const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
                        const uint32_t a6 = right_bot;
                        const uint32_t a7 = right_bot << 1;
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_right_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_right_mid_maj), "r"(c1_right_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val |= (uint64_t)result;
                        
                        c1_right_top_xor = c1_right_mid_xor;
                        c1_right_mid_xor = bottom_xor;
                        c1_right_top_maj = c1_right_mid_maj;
                        c1_right_mid_maj = bottom_maj;
                    }
                    
                    output[row-1][1] = output_val;
                }
            }
            //warpshiftStuff
        }
        

    }else{
        uint64_t r00 = warpStorage[warpId][(laneId * 8) - 2]
        uint64_t r01 = warpStorage[warpId][(laneId * 8) - 1]
        uint64_t r10 = warpStorage[warpId][(laneId * 8)]
        uint64_t r11 = warpStorage[warpId][(laneId * 8) + 1]
        uint64_t r20 = warpStorage[warpId][(laneId * 8) + 2]
        uint64_t r21 = warpStorage[warpId][(laneId * 8) + 3]
        uint64_t r30 = warpStorage[warpId][(laneId * 8) + 4]
        uint64_t r31 = warpStorage[warpId][(laneId * 8) + 5]
        uint64_t r40 = warpStorage[warpId][(laneId * 8) + 6]
        uint64_t r41 = warpStorage[warpId][(laneId * 8) + 7]
        uint64_t r50 = warpStorage[warpId][(laneId * 8) + 8]
        uint64_t r51 = warpStorage[warpId][(laneId * 8) + 9]

        for (int i = 0; i < 32; i++){
            // Storage organized as [row][column]
            uint64_t storage[6][2];
            storage[0][0] = r00; storage[0][1] = r01;
            storage[1][0] = r10; storage[1][1] = r11;
            storage[2][0] = r20; storage[2][1] = r21;
            storage[3][0] = r30; storage[3][1] = r31;
            storage[4][0] = r40; storage[4][1] = r41;
            storage[5][0] = r50; storage[5][1] = r51;
            
            uint64_t output[4][2];  // 4 output rows, 2 columns
            
            // Reuse chains for column 0
            uint32_t c0_left_top_xor, c0_left_mid_xor, c0_left_top_maj, c0_left_mid_maj;
            uint32_t c0_right_top_xor, c0_right_mid_xor, c0_right_top_maj, c0_right_mid_maj;
            
            // Reuse chains for column 1
            uint32_t c1_left_top_xor, c1_left_mid_xor, c1_left_top_maj, c1_left_mid_maj;
            uint32_t c1_right_top_xor, c1_right_mid_xor, c1_right_top_maj, c1_right_mid_maj;
            
            // Process 4 output rows
            #pragma unroll //encourage compiler to unroll for loop and put into thraed registers check via compiling with: nvcc -Xptxas -v kernel.cu -o kernel.o, and executing it, should have line ptxas info
            for (int row = 1; row <= 4; row++) {
                
                // ========== COLUMN 0 ==========
                {
                    // Split into 32-bit halves
                    uint32_t left_top = (uint32_t)(storage[row-1][0] >> 32);
                    uint32_t right_top = (uint32_t)(storage[row-1][0]);
                    uint32_t left_mid = (uint32_t)(storage[row][0] >> 32);
                    uint32_t right_mid = (uint32_t)(storage[row][0]);
                    uint32_t left_bot = (uint32_t)(storage[row+1][0] >> 32);
                    uint32_t right_bot = (uint32_t)(storage[row+1][0]);
                    
                    // For wrapping from column 1
                    uint32_t col1_left_top = (uint32_t)(storage[row-1][1] >> 32);
                    uint32_t col1_left_mid = (uint32_t)(storage[row][1] >> 32);
                    uint32_t col1_left_bot = (uint32_t)(storage[row+1][1] >> 32);
                    
                    uint64_t output_val = 0;
                    
                    // --- LEFT HALF (upper 32 bits) ---
                    {
                        const uint32_t a0 = left_top >> 1;
                        const uint32_t a1 = left_top;
                        const uint32_t a2 = (left_top << 1) | (right_top >> 31);
                        const uint32_t a3 = left_mid >> 1;
                        const uint32_t center = left_mid;
                        const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
                        const uint32_t a5 = left_bot >> 1;
                        const uint32_t a6 = left_bot;
                        const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        // Magic stage
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_left_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_left_mid_maj), "r"(c0_left_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val = ((uint64_t)result << 32);
                        
                        c0_left_top_xor = c0_left_mid_xor;
                        c0_left_mid_xor = bottom_xor;
                        c0_left_top_maj = c0_left_mid_maj;
                        c0_left_mid_maj = bottom_maj;
                    }
                    
                    // --- RIGHT HALF (lower 32 bits) ---
                    {
                        const uint32_t a0 = (left_top << 31) | (right_top >> 1);
                        const uint32_t a1 = right_top;
                        const uint32_t a2 = (right_top << 1) | (col1_left_top >> 31);  // Wrap to column 1
                        const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
                        const uint32_t center = right_mid;
                        const uint32_t a4 = (right_mid << 1) | (col1_left_mid >> 31);
                        const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
                        const uint32_t a6 = right_bot;
                        const uint32_t a7 = (right_bot << 1) | (col1_left_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_right_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_right_mid_maj), "r"(c0_right_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val |= (uint64_t)result;
                        
                        c0_right_top_xor = c0_right_mid_xor;
                        c0_right_mid_xor = bottom_xor;
                        c0_right_top_maj = c0_right_mid_maj;
                        c0_right_mid_maj = bottom_maj;
                    }
                    
                    output[row-1][0] = output_val;
                }
                
                // ========== COLUMN 1 ==========
                {
                    uint32_t left_top = (uint32_t)(storage[row-1][1] >> 32);
                    uint32_t right_top = (uint32_t)(storage[row-1][1]);
                    uint32_t left_mid = (uint32_t)(storage[row][1] >> 32);
                    uint32_t right_mid = (uint32_t)(storage[row][1]);
                    uint32_t left_bot = (uint32_t)(storage[row+1][1] >> 32);
                    uint32_t right_bot = (uint32_t)(storage[row+1][1]);
                    
                    // For wrapping from column 0
                    uint32_t col0_right_top = (uint32_t)(storage[row-1][0]);
                    uint32_t col0_right_mid = (uint32_t)(storage[row][0]);
                    uint32_t col0_right_bot = (uint32_t)(storage[row+1][0]);
                    
                    uint64_t output_val = 0;
                    
                    // --- LEFT HALF (upper 32 bits) ---
                    {
                        const uint32_t a0 = (col0_right_top << 1) | (left_top >> 31);  // Wrap from column 0
                        const uint32_t a1 = left_top;
                        const uint32_t a2 = (left_top << 1) | (right_top >> 31);
                        const uint32_t a3 = (col0_right_mid << 1) | (left_mid >> 31);
                        const uint32_t center = left_mid;
                        const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
                        const uint32_t a5 = (col0_right_bot << 1) | (left_bot >> 31);
                        const uint32_t a6 = left_bot;
                        const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_left_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_left_mid_maj), "r"(c1_left_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val = ((uint64_t)result << 32);
                        
                        c1_left_top_xor = c1_left_mid_xor;
                        c1_left_mid_xor = bottom_xor;
                        c1_left_top_maj = c1_left_mid_maj;
                        c1_left_mid_maj = bottom_maj;
                    }
                    
                    // --- RIGHT HALF (lower 32 bits) ---
                    {
                        const uint32_t a0 = (left_top << 31) | (right_top >> 1);
                        const uint32_t a1 = right_top;
                        const uint32_t a2 = right_top << 1;  // Wraps to column 0's left half
                        const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
                        const uint32_t center = right_mid;
                        const uint32_t a4 = right_mid << 1;
                        const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
                        const uint32_t a6 = right_bot;
                        const uint32_t a7 = right_bot << 1;
                        
                        if (row == 1) {
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                        }
                        
                        uint32_t bottom_xor, bottom_maj;
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                        
                        uint32_t magic0, magic1, magic2, result;
                        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_right_mid_xor), "r"(bottom_xor), "r"(center));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_right_mid_maj), "r"(c1_right_top_maj));
                        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                        
                        output_val |= (uint64_t)result;
                        
                        c1_right_top_xor = c1_right_mid_xor;
                        c1_right_mid_xor = bottom_xor;
                        c1_right_top_maj = c1_right_mid_maj;
                        c1_right_mid_maj = bottom_maj;
                    }
                    
                    output[row-1][1] = output_val;
                }
            }
            //warpshiftStuff
        }

        if ((laneId >= 8) && (laneId < 24)){
            uint64_t row1Final = r10 << 32 | r11 >> 32;
            uint64_t row2Final = r20 << 32 | r21 >> 32;
            uint64_t row3Final = r30 << 32 | r31 >> 32;
            uint64_t row4Final = r40 << 32 | r41 >> 32;
            globalData[((centralWarpStartY) * width) + globalX] = row1Final;
            globalData[((centralWarpStartY + 1) * width) + globalX] = row2Final;
            globalData[((centralWarpStartY + 2) * width) + globalX] = row3Final;
            globalData[((centralWarpStartY + 3) * width) + globalX] = row4Final;

        }
    }
}


void gol(uint64_t **current_ptr, uint64_t **next_ptr, int width, int height, int iterations) {
    uint64_t *current = *current_ptr;
    size_t size = (width / 64) * height * sizeof(uint64_t);

    uint64_t *d_current;
    cudaMalloc(&d_current, size);

    cudaMemcpy(d_current, *current_ptr, size, cudaMemcpyHostToDevice);

    dim3 blockDim(4, 128);
    dim3 gridDim(64,64);


    for (int iteration = 0; iteration < iterations; iteration+=32) {
        singleIteration<<<gridDim, blockDim>>>(d_current, height, arrayWidth);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(*current_ptr, d_current, size, cudaMemcpyDeviceToHost);

    cudaFree(d_current);
}