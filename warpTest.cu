#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>
#include <stdint.h>

struct Result8 {
    uint64_t store[4][2];
};

__device__ void print_binary64(uint64_t n) {
    for (int i = 63; i >= 0; i--) {
        uint64_t bit = (n >> i) & 1;
        printf("%" PRIu64, bit);
    }
    printf("\n");
}

__device__ void print_binary32(uint32_t n) {
    for (int i = 31; i >= 0; i--) {
        uint32_t bit = (n >> i) & 1;
        printf("%" PRIu32, bit);
    }
    printf("\n");
}

__global__ void  myKernel(){
    uint64_t storage[6][2];
    uint64_t r00 = 0x0000000000000000ULL;  // Row 0, left half
    uint64_t r01 = 0x0000000000000000ULL;  // Row 0, right half

    uint64_t r10 = 0x0000000000000000ULL;  // Row 1, left half
    uint64_t r11 = 0x0000000000000000ULL;  // Row 1, right half

    uint64_t r20 = 0x0000000000000E00ULL;  // Row 2, left half: bits 1,2,3 set
    uint64_t r21 = 0x0000000000000000ULL;  // Row 2, right half

    uint64_t r30 = 0x0000000000000000ULL;  // Row 3, left half
    uint64_t r31 = 0x0000000000000000ULL;  // Row 3, right half

    uint64_t r40 = 0x0000000000000000ULL;  // Row 4, left half
    uint64_t r41 = 0x0000000000000000ULL;  // Row 4, right half

    uint64_t r50 = 0x0000000000000000ULL;  // Row 5, left half
    uint64_t r51 = 0x0000000000000000ULL;  // Row 5, right half


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
    
    uint64_t output[4][2];  // 4 output rows, 2 columns
    //Result8 output;
    
    uint32_t top_0_xor, top_0_sum, mid_0_xor, mid_0_sum, bottom_0_xor, bottom_0_sum, top_1_xor, top_1_sum, mid_1_xor, mid_1_sum, bottom_1_xor, bottom_1_sum, top_2_xor, top_2_sum, mid_2_xor, mid_2_sum, bottom_2_xor, bottom_2_sum, top_3_xor, top_3_sum, mid_3_xor, mid_3_sum, bottom_3_xor, bottom_3_sum;
    uint32_t topLeft, top, topRight, left, middle, right, bottomLeft, bottom, bottomRight, twoCarry, twoSum, magic0, magic1, magic2;
    uint32_t left0, middle0, right0, left1, middle1, right1, left2, middle2, right2, left3, middle3, right3;
    uint32_t result0, result1, result2, result3;

    // Process 4 output rows
    #pragma unroll 4 //encourage compiler to unroll for loop (for all 4 iterations) and put into thraed registers check via compiling with: nvcc -Xptxas -v kernel.cu -o kernel.o, and executing it, should have line ptxas info
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

        output[row-1][0] = ((uint64_t)result0 << 32) | result1;
        output[row-1][1] = ((uint64_t)result2 << 32) | result3;
    }
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 2; col++) {
            printf("output[%d][%d] =", row, col);
            print_binary64(output[row][col]);
            printf("\n");
        }
    }
}

int main() {
    // Launch kernel with 1 block of 1 thread
    myKernel<<<1, 1>>>();

    // Wait for GPU to finish before exiting    return 0;
    cudaDeviceSynchronize();

    return 0;
}