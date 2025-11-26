// Grid is 16384 x 16384, each cell is one bit, each uint64_t holds 64 cells
// So the grid is represented as 256 x 16384 uint64_t elements
#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <time.h>
#include <inttypes.h>

struct Result8 {
    uint64_t store[4][2];
};

__device__ void print_binary64(uint64_t n) {
    for (int i = 63; i >= 0; i--) {
        uint64_t bit = (n >> i) & 1;
        printf("%" PRIu64, bit);
    }
}

__device__ __forceinline__ uint64_t readGlobalTimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Bit-parallel GOL implementation
__device__ __forceinline__ uint64_t rotate_left(uint64_t centre, uint64_t left){
    return (centre << 1) | (left >> 63);
}

__device__ __forceinline__ uint64_t rotate_right(uint64_t centre, uint64_t right){
    return (centre >> 1) | (right << 63);
}

__device__ __forceinline__ Result8 oneStepReduceSquareLOP3v1(uint64_t r00, uint64_t r01, uint64_t r10, uint64_t r11, uint64_t r20, uint64_t r21, uint64_t r30, uint64_t r31, uint64_t r40, uint64_t r41, uint64_t r50, uint64_t r51){
    uint64_t storage[6][2];

    storage[0][0] = r00;
    storage[0][1] = r01;
    storage[1][0] = r10;
    storage[1][1] = r11;
    storage[2][0] = r20;
    storage[2][1] = r21;
    storage[3][0] = r30;
    storage[3][1] = r31;
    storage[4][0] = r40;
    storage[4][1] = r41;
    storage[5][0] = r50;
    storage[5][1] = r51;
    
    //uint64_t output[4][2];  // 4 output rows, 2 columns
    Result8 output;
    
    uint32_t top_0_xor, top_0_sum, mid_0_xor, mid_0_sum, bottom_0_xor, bottom_0_sum, top_1_xor, top_1_sum, mid_1_xor, mid_1_sum, bottom_1_xor, bottom_1_sum, top_2_xor, top_2_sum, mid_2_xor, mid_2_sum, bottom_2_xor, bottom_2_sum, top_3_xor, top_3_sum, mid_3_xor, mid_3_sum, bottom_3_xor, bottom_3_sum;
    uint32_t topLeft, top, topRight, left, middle, right, bottomLeft, bottom, bottomRight, twoCarry, twoSum, magic0, magic1, magic2;
    uint32_t left0, middle0, right0, left1, middle1, right1, left2, middle2, right2, left3, middle3, right3;
    uint32_t result0, result1, result2, result3;

    // Process 4 output rows
    #pragma unroll //encourage compiler to unroll for loop (for all 4 iterations) and put into thraed registers check via compiling with: nvcc -Xptxas -v kernel.cu -o kernel.o, and executing it, should have line ptxas info
    for (int row = 1; row <= 4; row++) {
        if (row == 1){
            uint64_t topLeft0 = (r00 >> 1);
            uint64_t topLeft1 = (r01 >> 1) | (r00 << 63);
            uint64_t top0 = r00;
            uint64_t top1 = r01;
            uint64_t topRight0 = (r00 << 1) | (r01 >> 63); 
            uint64_t topRight1 = (r01 << 1);
            uint64_t midLeft0 = (r10 >> 1);
            uint64_t midLeft1 = (r11 >> 1) | (r10 << 63);
            uint64_t midRight0 = (r10 << 1) | (r11 >> 63);
            uint64_t midRight1 = (r11 << 1);

            // left (0)
            topLeft = (uint32_t)(topLeft0 >> 32);
            top = (uint32_t)(top0 >> 32);
            topRight = (uint32_t)(topRight0 >> 32);
            left = (uint32_t)(midLeft0 >> 32);
            left0 = left;
            middle = (uint32_t)(r10 >> 32);
            middle0 = middle;
            right = (uint32_t)(midRight0 >> 32);
            right0 = right;
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(top_0_xor) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(top_0_sum) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(mid_0_xor) : "r"(left), "r"(middle), "r"(right));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(mid_0_sum) : "r"(left), "r"(middle), "r"(right));

            // middle (1)
            topLeft = (uint32_t)(topLeft0);
            top = (uint32_t)(top0);
            topRight = (uint32_t)(topRight0);
            left = (uint32_t)(midLeft0);
            left1 = left;
            middle = (uint32_t)(r10);
            middle1 = middle;
            right = (uint32_t)(midRight0);
            right1 = right;
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(top_1_xor) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(top_1_sum) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(mid_1_xor) : "r"(left), "r"(middle), "r"(right));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(mid_1_sum) : "r"(left), "r"(middle), "r"(right));

            // middle (2)
            topLeft = (uint32_t)(topLeft1 >> 32);
            top = (uint32_t)(top1 >> 32);
            topRight = (uint32_t)(topRight1 >> 32);
            left = (uint32_t)(midLeft1 >> 32);
            left2 = left;
            middle = (uint32_t)(r11 >> 32);
            middle2 = middle;
            right = (uint32_t)(midRight1 >> 32);
            right2 = right;
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(top_2_xor) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(top_2_sum) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(mid_2_xor) : "r"(left), "r"(middle), "r"(right));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(mid_2_sum) : "r"(left), "r"(middle), "r"(right));

            //right (3)
            topLeft = (uint32_t)(topLeft1 >> 32);
            top = (uint32_t)(top1 >> 32);
            topRight = (uint32_t)(topRight1 >> 32);
            left = (uint32_t)(midLeft1 >> 32);
            left3 = left;
            middle = (uint32_t)(r11 >> 32);
            middle3 = middle;
            right = (uint32_t)(midRight1 >> 32);
            right3 = right;
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(top_3_xor) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(top_3_sum) : "r"(topLeft), "r"(top), "r"(topRight));
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(mid_3_xor) : "r"(left), "r"(middle), "r"(right));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(mid_3_sum) : "r"(left), "r"(middle), "r"(right));
        }

        uint64_t bottomLeft0 = (storage[row+1][0] >> 1);
        uint64_t bottomLeft1 = (storage[row+1][1] >> 1) | (storage[row+1][0] << 63);
        uint64_t bottom0 = storage[row+1][0];
        uint64_t bottom1 = storage[row+1][1];
        uint64_t bottomRight0 = (storage[row+1][0] << 1) | (storage[row+1][1] >> 63); 
        uint64_t bottomRight1 = (storage[row+1][1] << 1);


        //left (0) ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        bottomLeft = (uint32_t)(bottomLeft0 >> 32);
        bottom = (uint32_t)(bottom0 >> 32);
        bottomRight = (uint32_t)(bottomRight0 >> 32);

        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_0_xor) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_0_sum) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));


        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(twoCarry) : "r"(top_0_xor), "r"(left0), "r"(right0));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(twoSum) : "r"(top_0_xor), "r"(left0), "r"(right0));

        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(bottom_0_xor), "r"(twoCarry), "r"(middle0));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(middle0), "r"(twoSum));
        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(bottom_0_sum), "r"(top_0_sum));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result0) : "r"(magic2), "r"(magic0), "r"(magic1));

        top_0_xor = mid_0_xor;
        top_0_sum = mid_0_sum;
        mid_0_xor = bottom_0_xor;
        mid_0_sum = bottom_0_sum;
        left0 = bottomLeft;
        middle0 = bottom;
        right0 = bottomRight;

        //middle (1) ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        bottomLeft = (uint32_t)(bottomLeft0);
        bottom = (uint32_t)(bottom0);
        bottomRight = (uint32_t)(bottomRight0);
        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_1_xor) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_1_sum) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));

        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(twoCarry) : "r"(top_1_xor), "r"(left1), "r"(right1));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(twoSum) : "r"(top_1_xor), "r"(left1), "r"(right1));

        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(bottom_1_xor), "r"(twoCarry), "r"(middle1));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(middle1), "r"(twoSum));
        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(bottom_1_sum), "r"(top_1_sum));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result1) : "r"(magic2), "r"(magic0), "r"(magic1));

        top_1_xor = mid_1_xor;
        top_1_sum = mid_1_sum;
        mid_1_xor = bottom_1_xor;
        mid_1_sum = bottom_1_sum;
        left1 = bottomLeft;
        middle1 = bottom;
        right1 = bottomRight;

        //middle (2) ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        bottomLeft = (uint32_t)(bottomLeft1 >> 32);
        bottom = (uint32_t)(bottom1 >> 32);
        bottomRight = (uint32_t)(bottomRight1 >> 32);
        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_2_xor) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_2_sum) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));
        
        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(twoCarry) : "r"(top_2_xor), "r"(left2), "r"(right2));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(twoSum) : "r"(top_2_xor), "r"(left2), "r"(right2));

        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(bottom_2_xor), "r"(twoCarry), "r"(middle2));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(middle2), "r"(twoSum));
        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(bottom_2_sum), "r"(top_2_sum));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result2) : "r"(magic2), "r"(magic0), "r"(magic1));

        top_2_xor = mid_2_xor;
        top_2_sum = mid_2_sum;
        mid_2_xor = bottom_2_xor;
        mid_2_sum = bottom_2_sum;
        left2 = bottomLeft;
        middle2 = bottom;
        right2 = bottomRight;

        // right (3) ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        bottomLeft = (uint32_t)(bottomLeft1);
        bottom = (uint32_t)(bottom1);
        bottomRight = (uint32_t)(bottomRight1);
        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_3_xor) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_3_sum) : "r"(bottomLeft), "r"(bottom), "r"(bottomRight));

        asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(twoCarry) : "r"(top_3_xor), "r"(left3), "r"(right3));
        asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(twoSum) : "r"(top_3_xor), "r"(left3), "r"(right3));

        asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(bottom_3_xor), "r"(twoCarry), "r"(middle3));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(middle3), "r"(twoSum));
        asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(bottom_3_sum), "r"(top_3_sum));
        asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result3) : "r"(magic2), "r"(magic0), "r"(magic1));

        top_3_xor = mid_3_xor;
        top_3_sum = mid_3_sum;
        mid_3_xor = bottom_3_xor;
        mid_3_sum = bottom_3_sum;
        left3 = bottomLeft;
        middle3 = bottom;
        right3 = bottomRight;

        // output 64-bit results --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        output.store[row-1][0] = ((uint64_t)result0 << 32) | result1;
        output.store[row-1][1] = ((uint64_t)result2 << 32) | result3;
    }
    return output;
}

__device__ __forceinline__ Result8 oneStepReduceSquareLOP3v2(uint64_t r00, uint64_t r01, uint64_t r10, uint64_t r11, uint64_t r20, uint64_t r21, uint64_t r30, uint64_t r31, uint64_t r40, uint64_t r41, uint64_t r50, uint64_t r51){
    uint64_t storage[6][2];

    storage[0][0] = r00;
    storage[0][1] = r01;
    storage[1][0] = r10;
    storage[1][1] = r11;
    storage[2][0] = r20;
    storage[2][1] = r21;
    storage[3][0] = r30;
    storage[3][1] = r31;
    storage[4][0] = r40;
    storage[4][1] = r41;
    storage[5][0] = r50;
    storage[5][1] = r51;
    
    //uint64_t output[4][2];  // 4 output rows, 2 columns
    Result8 output;
    
    // Reuse chains for column 0
    uint32_t c0_left_top_xor, c0_left_mid_xor, c0_left_top_sum, c0_left_mid_sum;
    uint32_t c0_right_top_xor, c0_right_mid_xor, c0_right_top_sum, c0_right_mid_sum;
    
    // Reuse chains for column 1
    uint32_t c1_left_top_xor, c1_left_mid_xor, c1_left_top_sum, c1_left_mid_sum;
    uint32_t c1_right_top_xor, c1_right_mid_xor, c1_right_top_sum, c1_right_mid_sum;
    
    // Process 4 output rows
    #pragma unroll 4 //encourage compiler to unroll for loop (for all 4 iterations) and put into thraed registers check via compiling with: nvcc -Xptxas -v kernel.cu -o kernel.o, and executing it, should have line ptxas info
    for (int row = 1; row <= 4; row++) {
        
        // ========== COLUMN 0 (rx0) ==========
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
            
            // --- LEFT HALF (upper / left 32 bits) ---
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
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_top_sum) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_mid_sum) : "r"(a4), "r"(a3), "r"(center));
                }
                
                uint32_t c0_left_bot_xor, c0_left_bot_sum, c0_left_twoCarry, c0_left_twoSum;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_bot_xor) : "r"(a7), "r"(a6), "r"(a5));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_bot_sum) : "r"(a7), "r"(a6), "r"(a5));
                
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_twoCarry) : "r"(c0_left_top_xor), "r"(a4), "r"(a3));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_twoSum) : "r"(c0_left_top_xor), "r"(a4), "r"(a3));

                // Magic stage
                uint32_t magic0, magic1, magic2, result;
                asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_left_bot_xor), "r"(c0_left_twoCarry), "r"(center));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(c0_left_twoSum));
                asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_left_bot_sum), "r"(c0_left_top_sum));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
                output_val = ((uint64_t)result << 32);
                
                c0_left_top_xor = c0_left_mid_xor;
                c0_left_mid_xor = c0_left_bot_xor;
                c0_left_top_sum = c0_left_mid_sum;
                c0_left_mid_sum = c0_left_bot_sum;
            }
            
            // --- RIGHT HALF (lower / right 32 bits) ---
            {
                const uint32_t a0 = (right_top >> 1) | (left_top << 31);
                const uint32_t a1 = right_top;
                const uint32_t a2 = (right_top << 1) | (col1_left_top >> 31);  // Wrap to column 1
                const uint32_t a3 = (right_mid >> 1) | (left_mid << 31);
                const uint32_t center = right_mid;
                const uint32_t a4 = (right_mid << 1) | (col1_left_mid >> 31);
                const uint32_t a5 = (right_bot >> 1) | (left_bot << 31);
                const uint32_t a6 = right_bot;
                const uint32_t a7 = (right_bot << 1) | (col1_left_bot >> 31);
                
                if (row == 1) {
                    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_top_sum) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_mid_sum) : "r"(a4), "r"(a3), "r"(center));
                }
                
                uint32_t c0_right_bot_xor, c0_right_bot_sum, c0_right_twoCarry, c0_right_twoSum;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_bot_xor) : "r"(a7), "r"(a6), "r"(a5));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_bot_sum) : "r"(a7), "r"(a6), "r"(a5));
                
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_twoCarry) : "r"(c0_right_top_xor), "r"(a4), "r"(a3));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_twoSum) : "r"(c0_right_top_xor), "r"(a4), "r"(a3));
                
                uint32_t magic0, magic1, magic2, result;
                asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_right_bot_xor), "r"(c0_right_twoCarry), "r"(center));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(c0_right_twoSum));
                asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_right_bot_sum), "r"(c0_right_top_sum));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
                output_val |= (uint64_t)result;;    
                
                c0_right_top_xor = c0_right_mid_xor;
                c0_right_mid_xor = c0_right_bot_xor;
                c0_right_top_sum = c0_right_mid_sum;
                c0_right_mid_sum = c0_right_bot_sum;
            }
            
            output.store[row-1][0] = output_val;
        }
        
        // ========== COLUMN 1 (rx1) ==========
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
            
            // --- LEFT HALF (upper / left 32 bits) ---
            {
                const uint32_t a0 = (left_top >> 1) | (col0_right_top << 31);  // Wrap from column 0
                const uint32_t a1 = left_top;
                const uint32_t a2 = (left_top << 1) | (right_top >> 31);
                const uint32_t a3 = (left_mid >> 1) | (col0_right_mid << 31);
                const uint32_t center = left_mid;
                const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
                const uint32_t a5 = (left_bot >> 1) | (col0_right_bot << 31);
                const uint32_t a6 = left_bot;
                const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                
                if (row == 1) {
                    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_top_sum) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_mid_sum) : "r"(a4), "r"(a3), "r"(center));
                }
                
                uint32_t c1_left_bot_xor, c1_left_bot_sum, c1_left_twoCarry, c1_left_twoSum;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_bot_xor) : "r"(a7), "r"(a6), "r"(a5));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_bot_sum) : "r"(a7), "r"(a6), "r"(a5));
                
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_twoCarry) : "r"(c1_left_top_xor), "r"(a4), "r"(a3));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_twoSum) : "r"(c1_left_top_xor), "r"(a4), "r"(a3));
                
                uint32_t magic0, magic1, magic2, result;
                asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_left_bot_xor), "r"(c1_left_twoCarry), "r"(center));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(c1_left_twoSum));
                asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_left_bot_sum), "r"(c1_left_top_sum));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
                output_val = ((uint64_t)result << 32);
                
                c1_left_top_xor = c1_left_mid_xor;
                c1_left_mid_xor = c1_left_bot_xor;
                c1_left_top_sum = c1_left_mid_sum;
                c1_left_mid_sum = c1_left_bot_sum;
            }
            
            // --- RIGHT HALF (lower / right 32 bits) ---
            {
                const uint32_t a0 = (right_top >> 1) | (left_top << 31);
                const uint32_t a1 = right_top;
                const uint32_t a2 = right_top << 1;  // Wraps to column 0's left half
                const uint32_t a3 = (right_mid >> 1) | (left_mid << 31);
                const uint32_t center = right_mid;
                const uint32_t a4 = right_mid << 1;
                const uint32_t a5 = (right_bot >> 1) | (left_bot << 31);
                const uint32_t a6 = right_bot;
                const uint32_t a7 = right_bot << 1;
                
                if (row == 1) {
                    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_top_sum) : "r"(a2), "r"(a1), "r"(a0));
                    asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                    asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_mid_sum) : "r"(a4), "r"(a3), "r"(center));
                }
                
                uint32_t c1_right_bot_xor, c1_right_bot_sum, c1_right_twoCarry, c1_right_twoSum;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_bot_xor) : "r"(a7), "r"(a6), "r"(a5));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_bot_sum) : "r"(a7), "r"(a6), "r"(a5));
                
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_twoCarry) : "r"(c1_right_top_xor), "r"(a4), "r"(a3));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_twoSum) : "r"(c1_right_top_xor), "r"(a4), "r"(a3));
                
                uint32_t magic0, magic1, magic2, result;
                asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_right_bot_xor), "r"(c1_right_twoCarry), "r"(center));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(c1_right_twoSum));
                asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_right_bot_sum), "r"(c1_right_top_sum));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
                output_val |= (uint64_t)result;
                
                c1_right_top_xor = c1_right_mid_xor;
                c1_right_mid_xor = c1_right_bot_xor;
                c1_right_top_sum = c1_right_mid_sum;
                c1_right_mid_sum = c1_right_bot_sum;
            }
            
            output.store[row-1][1] = output_val;
        }
    }        
    return output;
}

__device__ __forceinline__ Result8 oneStepReduceSquareDoubleWord(uint64_t r00, uint64_t r01, uint64_t r10, uint64_t r11, uint64_t r20, uint64_t r21, uint64_t r30, uint64_t r31, uint64_t r40, uint64_t r41, uint64_t r50, uint64_t r51){
    uint64_t storage[6][2];

    storage[0][0] = r00;
    storage[0][1] = r01;
    storage[1][0] = r10;
    storage[1][1] = r11;
    storage[2][0] = r20;
    storage[2][1] = r21;
    storage[3][0] = r30;
    storage[3][1] = r31;
    storage[4][0] = r40;
    storage[4][1] = r41;
    storage[5][0] = r50;
    storage[5][1] = r51;
    
    //uint64_t output[4][2];  // 4 output rows, 2 columns
    Result8 output;
    #pragma unroll
    for (int row = 1; row < 5; row+=2){
        uint64_t topLeft = 0;
        uint64_t top = storage[row-1][0];
        uint64_t topRight = storage[row-1][1];
        uint64_t middle11 = 0;
        uint64_t middle12 = storage[row][0];
        uint64_t middle13 = storage[row][1];
        uint64_t middle21 = 0;
        uint64_t middle22 = storage[row+1][0];
        uint64_t middle23 = storage[row+1][1];
        uint64_t bottomLeft = 0;
        uint64_t bottom = storage[row+2][0];
        uint64_t bottomRight = storage[row+2][1];

        uint64_t tl = rotate_left(top, topLeft);
        uint64_t t = top;
        uint64_t tr = rotate_right(top, topRight);
        uint64_t m11 = rotate_left(middle12, middle11);
        uint64_t m12 = middle12;
        uint64_t m13 = rotate_right(middle12, middle13);
        uint64_t m21 = rotate_left(middle22, middle21);
        uint64_t m22 = middle22;
        uint64_t m23 = rotate_right(middle22, middle23);
        uint64_t bl = rotate_left(bottom, bottomLeft);
        uint64_t b = bottom;
        uint64_t br = rotate_right(bottom, bottomRight);

        // Half-Adders to sum the 8 neighbors:
        uint64_t sum1 = m13 ^ m23;
        uint64_t carry1 = m13 & m23;
        uint64_t sum2 = m11 ^ m21;
        uint64_t carry2 = m11 & m21;
        uint64_t sum3 = tl ^ t;
        uint64_t carry3 = tl & t;
        uint64_t sum4 = tr ^ m12;
        uint64_t carry4 = tr & m12;
        uint64_t sum5 = br ^ b;
        uint64_t carry5 = br & b;
        uint64_t sum6 = bl ^ m22;
        uint64_t carry6 = bl & m22;

        // Combining two Half-Adders into a full 3-bit addes with a 3-bit result:
        uint64_t bit00 = sum1 ^ sum2;
        uint64_t bit01 = (sum1 ^ sum2) ^ (carry1 ^ carry2);
        uint64_t bit02 = carry1 & carry2;

        uint64_t bit10 = sum3 ^ sum4;
        uint64_t bit11 = (sum3 ^ sum4) ^ (carry3 ^ carry4);
        uint64_t bit12 = carry3 & carry4;

        uint64_t bit20 = sum5 ^ sum6;
        uint64_t bit21 = (sum5 ^ sum6) ^ (carry5 ^ carry6);
        uint64_t bit22 = carry5 & carry6;

        // Add the three 3-bit numbers together to get final 4-bit neighbor counts, for both m12 and m22:
        uint64_t temp01 = bit00 & bit10;
        uint64_t temp02 = bit01 ^ bit11;
        uint64_t tempBit00 = bit00 ^ bit10;
        uint64_t tempBit01 = temp01 ^ temp02;
        uint64_t tempBit02 = bit02 | bit12 | (bit01 & bit11) | (temp01 & temp02); //tempBit02 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

        uint64_t temp11 = bit00 & bit20;
        uint64_t temp12 = bit01 ^ bit21;
        uint64_t tempBit10 = bit00 ^ bit20;
        uint64_t tempBit11 = temp11 ^ temp12;
        uint64_t tempBit12 = bit02 | bit22 | (bit01 & bit21) | (temp11 & temp12); //tempBit12 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

        // Final three bits for m12 and m22:
        uint64_t m12_temp = (~tempBit02) & tempBit01;
        uint64_t m12_count2 = (~tempBit00) & m12_temp; //is 1 if number of cells is equal to 2
        uint64_t m12_count3 = tempBit00 & m12_temp; //is 1 if number of cells is equal to 3

        uint64_t m22_temp = (~tempBit12) & tempBit11;
        uint64_t m22_count2 = (~tempBit10) & m22_temp;
        uint64_t m22_count3 = tempBit10 & m22_temp;

        // Apply GOL rules:
        output.store[row][0] = (m12 & m12_count2) | m12_count3;
        output.store[row+1][0] = (m22 & m22_count2) | m22_count3;
    }

    #pragma unroll
    for (int row = 1; row < 5; row+=2){
        uint64_t topLeft = storage[row-1][0];
        uint64_t top = storage[row-1][1];
        uint64_t topRight = 0;
        uint64_t middle11 = storage[row][0];
        uint64_t middle12 = storage[row][1];
        uint64_t middle13 = 0;
        uint64_t middle21 = storage[row+1][0];
        uint64_t middle22 = storage[row+1][1];
        uint64_t middle23 = 0;
        uint64_t bottomLeft = storage[row+2][0];
        uint64_t bottom = storage[row+2][1];
        uint64_t bottomRight = 0;

        uint64_t tl = rotate_left(top, topLeft);
        uint64_t t = top;
        uint64_t tr = rotate_right(top, topRight);
        uint64_t m11 = rotate_left(middle12, middle11);
        uint64_t m12 = middle12;
        uint64_t m13 = rotate_right(middle12, middle13);
        uint64_t m21 = rotate_left(middle22, middle21);
        uint64_t m22 = middle22;
        uint64_t m23 = rotate_right(middle22, middle23);
        uint64_t bl = rotate_left(bottom, bottomLeft);
        uint64_t b = bottom;
        uint64_t br = rotate_right(bottom, bottomRight);

        // Half-Adders to sum the 8 neighbors:
        uint64_t sum1 = m13 ^ m23;
        uint64_t carry1 = m13 & m23;
        uint64_t sum2 = m11 ^ m21;
        uint64_t carry2 = m11 & m21;
        uint64_t sum3 = tl ^ t;
        uint64_t carry3 = tl & t;
        uint64_t sum4 = tr ^ m12;
        uint64_t carry4 = tr & m12;
        uint64_t sum5 = br ^ b;
        uint64_t carry5 = br & b;
        uint64_t sum6 = bl ^ m22;
        uint64_t carry6 = bl & m22;

        // Combining two Half-Adders into a full 3-bit addes with a 3-bit result:
        uint64_t bit00 = sum1 ^ sum2;
        uint64_t bit01 = (sum1 ^ sum2) ^ (carry1 ^ carry2);
        uint64_t bit02 = carry1 & carry2;

        uint64_t bit10 = sum3 ^ sum4;
        uint64_t bit11 = (sum3 ^ sum4) ^ (carry3 ^ carry4);
        uint64_t bit12 = carry3 & carry4;

        uint64_t bit20 = sum5 ^ sum6;
        uint64_t bit21 = (sum5 ^ sum6) ^ (carry5 ^ carry6);
        uint64_t bit22 = carry5 & carry6;

        // Add the three 3-bit numbers together to get final 4-bit neighbor counts, for both m12 and m22:
        uint64_t temp01 = bit00 & bit10;
        uint64_t temp02 = bit01 ^ bit11;
        uint64_t tempBit00 = bit00 ^ bit10;
        uint64_t tempBit01 = temp01 ^ temp02;
        uint64_t tempBit02 = bit02 | bit12 | (bit01 & bit11) | (temp01 & temp02); //tempBit02 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

        uint64_t temp11 = bit00 & bit20;
        uint64_t temp12 = bit01 ^ bit21;
        uint64_t tempBit10 = bit00 ^ bit20;
        uint64_t tempBit11 = temp11 ^ temp12;
        uint64_t tempBit12 = bit02 | bit22 | (bit01 & bit21) | (temp11 & temp12); //tempBit12 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

        // Final three bits for m12 and m22:
        uint64_t m12_temp = (~tempBit02) & tempBit01;
        uint64_t m12_count2 = (~tempBit00) & m12_temp; //is 1 if number of cells is equal to 2
        uint64_t m12_count3 = tempBit00 & m12_temp; //is 1 if number of cells is equal to 3

        uint64_t m22_temp = (~tempBit12) & tempBit11;
        uint64_t m22_count2 = (~tempBit10) & m22_temp;
        uint64_t m22_count3 = tempBit10 & m22_temp;

        // Apply GOL rules:
        output.store[row][1] = (m12 & m12_count2) | m12_count3;
        output.store[row+1][1] = (m22 & m22_count2) | m22_count3;
    }
    return output;
}

__device__ __forceinline__ Result8 oneStepReduceSquareSingleWord(uint64_t r00, uint64_t r01, uint64_t r10, uint64_t r11, uint64_t r20, uint64_t r21, uint64_t r30, uint64_t r31, uint64_t r40, uint64_t r41, uint64_t r50, uint64_t r51){
    uint64_t storage[6][2];

    storage[0][0] = r00;
    storage[0][1] = r01;
    storage[1][0] = r10;
    storage[1][1] = r11;
    storage[2][0] = r20;
    storage[2][1] = r21;
    storage[3][0] = r30;
    storage[3][1] = r31;
    storage[4][0] = r40;
    storage[4][1] = r41;
    storage[5][0] = r50;
    storage[5][1] = r51;
    
    //uint64_t output[4][2];  // 4 output rows, 2 columns
    Result8 output;
    #pragma unroll
    for (int row = 1; row < 5; row++){
        uint64_t topLeft = 0;
        uint64_t top = storage[row-1][0];
        uint64_t topRight = storage[row-1][1];
        uint64_t left = 0;
        uint64_t center = storage[row][0];
        uint64_t right = storage[row][1];
        uint64_t bottomLeft = 0;
        uint64_t bottom = storage[row+1][0];
        uint64_t bottomRight = storage[row+1][1];

        uint64_t tl = rotate_left(top, topLeft);
        uint64_t t = top;
        uint64_t tr = rotate_right(top, topRight);
        uint64_t l = rotate_left(center, left);
        uint64_t r = rotate_right(center, right);
        uint64_t bl = rotate_left(bottom, bottomLeft);
        uint64_t b = bottom;
        uint64_t br = rotate_right(bottom, bottomRight);

        //printf("center:\n");
        //print_binary(center);

        uint64_t sum1 = tl ^ t;
        uint64_t carry1 = tl & t;

        uint64_t sum2 = tr ^ r;
        uint64_t carry2 = tr & r;

        uint64_t sum3 = br ^ b;
        uint64_t carry3 = br & b;

        uint64_t sum4 = bl ^ l;
        uint64_t carry4 = bl & l;

        uint64_t bit00 = sum1 ^ sum2;
        uint64_t bit01 = (sum1 & sum2) ^ (carry1 ^ carry2);
        uint64_t bit02 = (carry1 & carry2);

        uint64_t bit10 = sum3 ^ sum4;
        uint64_t bit11 = (sum3 & sum4) ^ (carry3 ^ carry4);
        uint64_t bit12 = (carry3 & carry4);

        uint64_t temp1 = bit00 & bit10;
        uint64_t temp2 = bit01 ^ bit11;
        uint64_t tempBit0 = bit00 ^ bit10;
        uint64_t tempBit1 = temp1 ^ temp2;
        uint64_t tempBit2 = bit02 | bit12 | (bit01 & bit11) | (temp1 & temp2);


        //above just sums the number of ones, and represnts it in three 64 bit numbers, one bit in each number
        uint64_t finalTemp = (~tempBit2) & tempBit1;
        uint64_t two = (~tempBit0) & finalTemp;
        uint64_t three = tempBit0 & finalTemp;
        output.store[row-1][0] = three | (two & center);


        topLeft = storage[row-1][0];
        top = storage[row-1][1];
        topRight = 0;
        left = storage[row][0];
        center = storage[row][1];
        right = 0;
        bottomLeft = storage[row+1][0];
        bottom = storage[row+1][1];
        bottomRight = 0;

        tl = rotate_left(top, topLeft);
        t = top;
        tr = rotate_right(top, topRight);
        l = rotate_left(center, left);
        r = rotate_right(center, right);
        bl = rotate_left(bottom, bottomLeft);
        b = bottom;
        br = rotate_right(bottom, bottomRight);

        //printf("center:\n");
        //print_binary(center);

        sum1 = tl ^ t;
        carry1 = tl & t;

        sum2 = tr ^ r;
        carry2 = tr & r;

        sum3 = br ^ b;
        carry3 = br & b;

        sum4 = bl ^ l;
        carry4 = bl & l;

        bit00 = sum1 ^ sum2;
        bit01 = (sum1 & sum2) ^ (carry1 ^ carry2);
        bit02 = (carry1 & carry2);

        bit10 = sum3 ^ sum4;
        bit11 = (sum3 & sum4) ^ (carry3 ^ carry4);
        bit12 = (carry3 & carry4);

        temp1 = bit00 & bit10;
        temp2 = bit01 ^ bit11;
        tempBit0 = bit00 ^ bit10;
        tempBit1 = temp1 ^ temp2;
        tempBit2 = bit02 | bit12 | (bit01 & bit11) | (temp1 & temp2);


        //above just sums the number of ones, and represnts it in three 64 bit numbers, one bit in each number
        finalTemp = (~tempBit2) & tempBit1;
        two = (~tempBit0) & finalTemp;
        three = tempBit0 & finalTemp;
        output.store[row-1][1] = three | (two & center);
    }
    return output;
}

__global__ void multistepKernel(uint64_t* globalData, int height, int width, int iteration) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned long long int totalTime = 0;
    unsigned long long int totalTime2 = 0;
    unsigned long long int totalTime3 = 0;
    unsigned long long int t20 = clock64();
    // Shared memory for 16 warps, each warp has 256 uint64_t (128 rows × 2 columns)
    __shared__ uint64_t warpStorage[16][256];
    // uint64_t totalTime = 0;
    // Thread/block coordinates

    // Warp identification
    int warpId = (ty >> 5) + (tx * 4);  // 16 warps per block
    int laneId = ty & 31;  // 31 is 0x1F, which is 2^5 - 1 (binary: 11111)
    // 0–31 within the warp

    // Global grid position
    int globalX = (blockIdx.x * blockDim.x) + tx;
    int globalXleft  = (globalX - 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)
    int globalXright = (globalX + 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)
    
    unsigned long long int t0 = clock64();

    float yQuarter = (ty / 32) * 0.25f;
    unsigned long long int t1 = clock64();
    totalTime += t1 - t0;

    int centralWarpStartY = (blockIdx.y * 256) + (256 * yQuarter);
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
    uint64_t r00;
    uint64_t r01;
    uint64_t r50;
    uint64_t r51;
    if (laneId == 0){
        r00 = 0ULL;
        r01 = 0ULL;
    } else{
        r00 = warpStorage[warpId][(laneId * 8) - 2];
        r01 = warpStorage[warpId][(laneId * 8) - 1];
    }

    if (laneId == 31){
        r50 = 0ULL;
        r51 = 0ULL;
    }else{
        r50 = warpStorage[warpId][(laneId * 8) + 8];
        r51 = warpStorage[warpId][(laneId * 8) + 9];
    }
    uint64_t r10 = warpStorage[warpId][(laneId * 8)];
    uint64_t r11 = warpStorage[warpId][(laneId * 8) + 1];
    uint64_t r20 = warpStorage[warpId][(laneId * 8) + 2];
    uint64_t r21 = warpStorage[warpId][(laneId * 8) + 3];
    uint64_t r30 = warpStorage[warpId][(laneId * 8) + 4];
    uint64_t r31 = warpStorage[warpId][(laneId * 8) + 5];
    uint64_t r40 = warpStorage[warpId][(laneId * 8) + 6];
    uint64_t r41 = warpStorage[warpId][(laneId * 8) + 7];
    unsigned long long int t30 = clock64();
    #pragma unroll
    for (int i = 0; i < 32; i++){
        //uint64_t t0 = readGlobalTimer();
        // unsigned long long int t0 = clock64();
        unsigned long long int t30 = clock64();
        Result8 r = oneStepReduceSquareLOP3v2(r00,r01,r10,r11,r20,r21,r30,r31,r40,r41,r50,r51);
        unsigned long long int t31 = clock64();
        totalTime3+= t31 - t30;
        // unsigned long long int t1 = clock64();
        // totalTime += t1-t0;
        //uint64_t t1 = readGlobalTimer();
        //totalTime += t1 - t0;
        r10 = r.store[0][0];
        r11 = r.store[0][1];
        r20 = r.store[1][0];
        r21 = r.store[1][1];
        r30 = r.store[2][0];
        r31 = r.store[2][1];
        r40 = r.store[3][0];
        r41 = r.store[3][1];

        // warpshiftStuff
        // unsigned long long int t0 = clock64();
        // Split into 32-bit parts
        uint32_t r10_lo = (uint32_t)(r10 & 0xFFFFFFFF);
        uint32_t r10_hi = (uint32_t)(r10 >> 32);
        uint32_t r11_lo = (uint32_t)(r11 & 0xFFFFFFFF);
        uint32_t r11_hi = (uint32_t)(r11 >> 32);
        uint32_t r40_lo = (uint32_t)(r40 & 0xFFFFFFFF);
        uint32_t r40_hi = (uint32_t)(r40 >> 32);
        uint32_t r41_lo = (uint32_t)(r41 & 0xFFFFFFFF);
        uint32_t r41_hi = (uint32_t)(r41 >> 32);

        // Pass r10/r11 UP (to laneId-1) and receive from above
        uint32_t recv_r50_lo = __shfl_up_sync(0xffffffff, r10_lo, 1);
        uint32_t recv_r50_hi = __shfl_up_sync(0xffffffff, r10_hi, 1);
        uint32_t recv_r51_lo = __shfl_up_sync(0xffffffff, r11_lo, 1);
        uint32_t recv_r51_hi = __shfl_up_sync(0xffffffff, r11_hi, 1);

        // Pass r40/r41 DOWN (to laneId+1) and receive from below
        uint32_t recv_r00_lo = __shfl_down_sync(0xffffffff, r40_lo, 1);
        uint32_t recv_r00_hi = __shfl_down_sync(0xffffffff, r40_hi, 1);
        uint32_t recv_r01_lo = __shfl_down_sync(0xffffffff, r41_lo, 1);
        uint32_t recv_r01_hi = __shfl_down_sync(0xffffffff, r41_hi, 1);
        // Reconstruct received values
        r00 = ((uint64_t)recv_r00_hi << 32) | recv_r00_lo;
        r01 = ((uint64_t)recv_r01_hi << 32) | recv_r01_lo;
        r50 = ((uint64_t)recv_r50_hi << 32) | recv_r50_lo;
        r51 = ((uint64_t)recv_r51_hi << 32) | recv_r51_lo;
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
    unsigned long long int t21 = clock64();
    totalTime2 += t21 - t20;
}

extern "C" void gol(uint64_t *current_ptr, int width, int height, int iterations) {
    size_t size = width * height * sizeof(uint64_t);
    uint64_t *d_current;
    cudaMalloc(&d_current, size);

    cudaMemcpy(d_current, current_ptr, size, cudaMemcpyHostToDevice);

    dim3 blockDim(4, 128);
    dim3 gridDim(64,64);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int iteration = 0; iteration < iterations; iteration+=32) {
        multistepKernel<<<gridDim, blockDim>>>(d_current, height, width, iteration);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
   
    printf("Completed in %.2f ms\n", milliseconds);
    cudaMemcpy(current_ptr, d_current, size, cudaMemcpyDeviceToHost);

    cudaFree(d_current);
}

// nvcc -O3 -arch=sm_86 -Xptxas -fopenmp -v warpHalo.cu -o gol_sim

//