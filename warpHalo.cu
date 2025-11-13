// Grid is 16384 x 16384, each cell is one bit, each uint64_t holds 64 cells
// So the grid is represented as 256 x 16384 uint64_t elements
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Bit-parallel GOL implementation
__device__ inline uint64_t rotate_left(uint64_t centre, uint64_t left){
    return (centre << 1) | (left >> 63);
}

__device__ inline uint64_t rotate_right(uint64_t centre, uint64_t right){
    return (centre >> 1) | (right << 63);
}

// __device__ inline void RowDoubleLeftUpdate(uint64_t *nextTop, uint64_t *nextBottom, uint64_t topLeft,uint64_t top,uint64_t topRight,uint64_t middle11, uint64_t middle12,uint64_t middle13,uint64_t middle 21, uint64_t middle22,uint64_t middle23,uint64_t bottomLeft,uint64_t bottom,uint64_t bottomRight) {
//     uint64_t tl = topLeft << 1;
//     uint64_t t = topLeft;
//     uint64_t tr = rotate_right(topLeft, top);
//     uint64_t m11 = middle11 << 1;
//     uint64_t m12 = middle11;
//     uint64_t m13 = rotate_right(middle11, middle12);
//     uint64_t m21 = middle21 << 1;
//     uint64_t m22 = middle21;
//     uint64_t m23 = rotate_right(middle21, middle22);
//     uint64_t bl = bottomLeft << 1;
//     uint64_t b = bottomLeft;
//     uint64_t br = rotate_right(bottomLeft, bottom);
    
    
//     // Half-Adders to sum the 8 neighbors:
//     uint64_t sum1 = m13 ^ m23;
//     uint64_t carry1 = m13 & m23;
//     uint64_t sum2 = m11 ^ m21;
//     uint64_t carry2 = m11 & m21;
//     uint64_t sum3 = tl ^ t;
//     uint64_t carry3 = tl & t;
//     uint64_t sum4 = tr ^ m12;
//     uint64_t carry4 = tr & m12;
//     uint64_t sum5 = br ^ b;
//     uint64_t carry5 = br & b;
//     uint64_t sum6 = bl ^ m22;
//     uint64_t carry6 = bl & m22;

//     // Combining two Half-Adders into a full 3-bit addes with a 3-bit result:
//     uint64_t bit00 = sum1 ^ sum2;
//     uint64_t bit01 = (sum1 ^ sum2) ^ (carry1 ^ carry2);
//     uint64_t bit02 = carry1 & carry2;

//     uint64_t bit10 = sum3 ^ sum4;
//     uint64_t bit11 = (sum3 ^ sum4) ^ (carry3 ^ carry4);
//     uint64_t bit12 = carry3 & carry4;

//     uint64_t bit20 = sum5 ^ sum6;
//     uint64_t bit21 = (sum5 ^ sum6) ^ (carry5 ^ carry6);
//     uint64_t bit22 = carry5 & carry6;

//     // Add the three 3-bit numbers together to get final 4-bit neighbor counts, for both m12 and m22:
//     uint64_t temp01 = bit00 & bit10;
//     uint64_t temp02 = bit01 ^ bit11;
//     uint64_t tempBit00 = bit00 ^ bit10;
//     uint64_t tempBit01 = temp01 ^ temp02;
//     uint64_t tempBit02 = bit02 | bit12 | (bit01 & bit11) | (temp01 & temp02); //tempBit02 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

//     uint64_t temp11 = bit00 & bit20;
//     uint64_t temp12 = bit01 ^ bit21;
//     uint64_t tempBit10 = bit00 ^ bit20;
//     uint64_t tempBit11 = temp11 ^ temp12;
//     uint64_t tempBit12 = bit02 | bit22 | (bit01 & bit21) | (temp11 & temp12); //tempBit12 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

//     // Final three bits for m12 and m22:
//     uint64_t m12_temp = (~tempBit02) & tempBit01;
//     uint64_t m12_count2 = (~tempBit00) & m12_temp; //is 1 if number of cells is equal to 2
//     uint64_t m12_count3 = tempBit00 & m12_temp; //is 1 if number of cells is equal to 3

//     uint64_t m22_temp = (~tempBit12) & tempBit11;
//     uint64_t m22_count2 = (~tempBit10) & m22_temp;
//     uint64_t m22_count3 = tempBit10 & m22_temp;

//     // Apply GOL rules:
//     *nextTop = (m11 & m12_count2) | m12_count3;
//     *nextBottom = (m21 & m22_count2) | m22_count3;
// }

// __device__ inline void RowDoubleRightUpdate(uint64_t *nextTop, uint64_t *nextBottom, uint64_t topLeft,uint64_t top,uint64_t topRight,uint64_t middle11, uint64_t middle12,uint64_t middle13,uint64_t middle 21, uint64_t middle22,uint64_t middle23,uint64_t bottomLeft,uint64_t bottom,uint64_t bottomRight) {
//     uint64_t tl = rotate_left(topRight, top);
//     uint64_t t = topRight;
//     uint64_t tr = topRight >> 1;
//     uint64_t m11 = rotate_left(middle13, middle12);
//     uint64_t m12 = middle13;
//     uint64_t m13 = middle13 >> 1;
//     uint64_t m21 = rotate_left(middle23, middle22);
//     uint64_t m22 = middle23;
//     uint64_t m23 = middle23 >> 1;
//     uint64_t bl = rotate_left(bottomRight, bottom);
//     uint64_t b = bottomRight;
//     uint64_t br = bottomRight >> 1;
    
//     // Half-Adders to sum the 8 neighbors:
//     uint64_t sum1 = m13 ^ m23;
//     uint64_t carry1 = m13 & m23;
//     uint64_t sum2 = m11 ^ m21;
//     uint64_t carry2 = m11 & m21;
//     uint64_t sum3 = tl ^ t;
//     uint64_t carry3 = tl & t;
//     uint64_t sum4 = tr ^ m12;
//     uint64_t carry4 = tr & m12;
//     uint64_t sum5 = br ^ b;
//     uint64_t carry5 = br & b;
//     uint64_t sum6 = bl ^ m22;
//     uint64_t carry6 = bl & m22;

//     // Combining two Half-Adders into a full 3-bit addes with a 3-bit result:
//     uint64_t bit00 = sum1 ^ sum2;
//     uint64_t bit01 = (sum1 ^ sum2) ^ (carry1 ^ carry2);
//     uint64_t bit02 = carry1 & carry2;

//     uint64_t bit10 = sum3 ^ sum4;
//     uint64_t bit11 = (sum3 ^ sum4) ^ (carry3 ^ carry4);
//     uint64_t bit12 = carry3 & carry4;

//     uint64_t bit20 = sum5 ^ sum6;
//     uint64_t bit21 = (sum5 ^ sum6) ^ (carry5 ^ carry6);
//     uint64_t bit22 = carry5 & carry6;

//     // Add the three 3-bit numbers together to get final 4-bit neighbor counts, for both m12 and m22:
//     uint64_t temp01 = bit00 & bit10;
//     uint64_t temp02 = bit01 ^ bit11;
//     uint64_t tempBit00 = bit00 ^ bit10;
//     uint64_t tempBit01 = temp01 ^ temp02;
//     uint64_t tempBit02 = bit02 | bit12 | (bit01 & bit11) | (temp01 & temp02); //tempBit02 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

//     uint64_t temp11 = bit00 & bit20;
//     uint64_t temp12 = bit01 ^ bit21;
//     uint64_t tempBit10 = bit00 ^ bit20;
//     uint64_t tempBit11 = temp11 ^ temp12;
//     uint64_t tempBit12 = bit02 | bit22 | (bit01 & bit21) | (temp11 & temp12); //tempBit12 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

//     // Final three bits for m12 and m22:
//     uint64_t m12_temp = (~tempBit02) & tempBit01;
//     uint64_t m12_count2 = (~tempBit00) & m12_temp; //is 1 if number of cells is equal to 2
//     uint64_t m12_count3 = tempBit00 & m12_temp; //is 1 if number of cells is equal to 3

//     uint64_t m22_temp = (~tempBit12) & tempBit11;
//     uint64_t m22_count2 = (~tempBit10) & m22_temp;
//     uint64_t m22_count3 = tempBit10 & m22_temp;

//     // Apply GOL rules:
//     *nextTop = (m13 & m12_count2) | m12_count3;
//     *nextBottom = (m23 & m22_count2) | m22_count3;
// }

// __device__ inline void RowDoubleCentralUpdate(uint64_t *nextTop, uint64_t *nextBottom, uint64_t topLeft,uint64_t top,uint64_t topRight,uint64_t middle11, uint64_t middle12,uint64_t middle13,uint64_t middle 21, uint64_t middle22,uint64_t middle23,uint64_t bottomLeft,uint64_t bottom,uint64_t bottomRight) {
//     uint64_t tl = rotate_left(top, topLeft);
//     uint64_t t = top;
//     uint64_t tr = rotate_right(top, topRight);
//     uint64_t m11 = rotate_left(middle12, middle11);
//     uint64_t m12 = middle12;
//     uint64_t m13 = rotate_right(middle12, middle13);
//     uint64_t m21 = rotate_left(middle22, middle21);
//     uint64_t m22 = middle22;
//     uint64_t m23 = rotate_right(middle22, middle23);
//     uint64_t bl = rotate_left(bottom, bottomLeft);
//     uint64_t b = bottom;
//     uint64_t br = rotate_right(bottom, bottomRight);
    
//     // Half-Adders to sum the 8 neighbors:
//     uint64_t sum1 = m13 ^ m23;
//     uint64_t carry1 = m13 & m23;
//     uint64_t sum2 = m11 ^ m21;
//     uint64_t carry2 = m11 & m21;
//     uint64_t sum3 = tl ^ t;
//     uint64_t carry3 = tl & t;
//     uint64_t sum4 = tr ^ m12;
//     uint64_t carry4 = tr & m12;
//     uint64_t sum5 = br ^ b;
//     uint64_t carry5 = br & b;
//     uint64_t sum6 = bl ^ m22;
//     uint64_t carry6 = bl & m22;

//     // Combining two Half-Adders into a full 3-bit addes with a 3-bit result:
//     uint64_t bit00 = sum1 ^ sum2;
//     uint64_t bit01 = (sum1 ^ sum2) ^ (carry1 ^ carry2);
//     uint64_t bit02 = carry1 & carry2;

//     uint64_t bit10 = sum3 ^ sum4;
//     uint64_t bit11 = (sum3 ^ sum4) ^ (carry3 ^ carry4);
//     uint64_t bit12 = carry3 & carry4;

//     uint64_t bit20 = sum5 ^ sum6;
//     uint64_t bit21 = (sum5 ^ sum6) ^ (carry5 ^ carry6);
//     uint64_t bit22 = carry5 & carry6;

//     // Add the three 3-bit numbers together to get final 4-bit neighbor counts, for both m12 and m22:
//     uint64_t temp01 = bit00 & bit10;
//     uint64_t temp02 = bit01 ^ bit11;
//     uint64_t tempBit00 = bit00 ^ bit10;
//     uint64_t tempBit01 = temp01 ^ temp02;
//     uint64_t tempBit02 = bit02 | bit12 | (bit01 & bit11) | (temp01 & temp02); //tempBit02 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

//     uint64_t temp11 = bit00 & bit20;
//     uint64_t temp12 = bit01 ^ bit21;
//     uint64_t tempBit10 = bit00 ^ bit20;
//     uint64_t tempBit11 = temp11 ^ temp12;
//     uint64_t tempBit12 = bit02 | bit22 | (bit01 & bit21) | (temp11 & temp12); //tempBit12 = 1 if count >= 4, i.e. bit2=1 (not overflow) or bit3=1 (overflow)

//     // Final three bits for m12 and m22:
//     uint64_t m12_temp = (~tempBit02) & tempBit01;
//     uint64_t m12_count2 = (~tempBit00) & m12_temp; //is 1 if number of cells is equal to 2
//     uint64_t m12_count3 = tempBit00 & m12_temp; //is 1 if number of cells is equal to 3

//     uint64_t m22_temp = (~tempBit12) & tempBit11;
//     uint64_t m22_count2 = (~tempBit10) & m22_temp;
//     uint64_t m22_count3 = tempBit10 & m22_temp;

//     // Apply GOL rules:
//     *nextTop = (m12 & m12_count2) | m12_count3;
//     *nextBottom = (m22 & m22_count2) | m22_count3;
// }

// __device__ inline void fourByFourCentralUpdate(
//     uint64_t r00, uint64_t r01,  // Row 0 (top halo)
//     uint64_t r10, uint64_t r11,  // Row 1 (output)
//     uint64_t r20, uint64_t r21,  // Row 2 (output)
//     uint64_t r30, uint64_t r31,  // Row 3 (output)
//     uint64_t r40, uint64_t r41,  // Row 4 (output)
//     uint64_t r50, uint64_t r51,  // Row 5 (bottom halo)
//     uint64_t& out10, uint64_t& out11,  // Output row 1
//     uint64_t& out20, uint64_t& out21,  // Output row 2
//     uint64_t& out30, uint64_t& out31,  // Output row 3
//     uint64_t& out40, uint64_t& out41   // Output row 4
// ) {
//     // Storage organized as [row][column]
//     uint64_t storage[6][2];
//     storage[0][0] = r00; storage[0][1] = r01;
//     storage[1][0] = r10; storage[1][1] = r11;
//     storage[2][0] = r20; storage[2][1] = r21;
//     storage[3][0] = r30; storage[3][1] = r31;
//     storage[4][0] = r40; storage[4][1] = r41;
//     storage[5][0] = r50; storage[5][1] = r51;
    
//     uint64_t output[4][2];  // 4 output rows, 2 columns
    
//     // Reuse chains for column 0
//     uint32_t c0_left_top_xor, c0_left_mid_xor, c0_left_top_maj, c0_left_mid_maj;
//     uint32_t c0_right_top_xor, c0_right_mid_xor, c0_right_top_maj, c0_right_mid_maj;
    
//     // Reuse chains for column 1
//     uint32_t c1_left_top_xor, c1_left_mid_xor, c1_left_top_maj, c1_left_mid_maj;
//     uint32_t c1_right_top_xor, c1_right_mid_xor, c1_right_top_maj, c1_right_mid_maj;
    
//     // Process 4 output rows
//     #pragma unroll //encourage compiler to unroll for loop and put into thraed registers check via compiling with: nvcc -Xptxas -v kernel.cu -o kernel.o, and executing it, should have line ptxas info
//     for (int row = 1; row <= 4; row++) {
        
//         // ========== COLUMN 0 ==========
//         {
//             // Split into 32-bit halves
//             uint32_t left_top = (uint32_t)(storage[row-1][0] >> 32);
//             uint32_t right_top = (uint32_t)(storage[row-1][0]);
//             uint32_t left_mid = (uint32_t)(storage[row][0] >> 32);
//             uint32_t right_mid = (uint32_t)(storage[row][0]);
//             uint32_t left_bot = (uint32_t)(storage[row+1][0] >> 32);
//             uint32_t right_bot = (uint32_t)(storage[row+1][0]);
            
//             // For wrapping from column 1
//             uint32_t col1_left_top = (uint32_t)(storage[row-1][1] >> 32);
//             uint32_t col1_left_mid = (uint32_t)(storage[row][1] >> 32);
//             uint32_t col1_left_bot = (uint32_t)(storage[row+1][1] >> 32);
            
//             uint64_t output_val = 0;
            
//             // --- LEFT HALF (upper 32 bits) ---
//             {
//                 const uint32_t a0 = left_top >> 1;
//                 const uint32_t a1 = left_top;
//                 const uint32_t a2 = (left_top << 1) | (right_top >> 31);
//                 const uint32_t a3 = left_mid >> 1;
//                 const uint32_t center = left_mid;
//                 const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
//                 const uint32_t a5 = left_bot >> 1;
//                 const uint32_t a6 = left_bot;
//                 const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                
//                 if (row == 1) {
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
//                 }
                
//                 uint32_t bottom_xor, bottom_maj;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                
//                 // Magic stage
//                 uint32_t magic0, magic1, magic2, result;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_left_mid_xor), "r"(bottom_xor), "r"(center));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_left_mid_maj), "r"(c0_left_top_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
//                 output_val = ((uint64_t)result << 32);
                
//                 c0_left_top_xor = c0_left_mid_xor;
//                 c0_left_mid_xor = bottom_xor;
//                 c0_left_top_maj = c0_left_mid_maj;
//                 c0_left_mid_maj = bottom_maj;
//             }
            
//             // --- RIGHT HALF (lower 32 bits) ---
//             {
//                 const uint32_t a0 = (left_top << 31) | (right_top >> 1);
//                 const uint32_t a1 = right_top;
//                 const uint32_t a2 = (right_top << 1) | (col1_left_top >> 31);  // Wrap to column 1
//                 const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
//                 const uint32_t center = right_mid;
//                 const uint32_t a4 = (right_mid << 1) | (col1_left_mid >> 31);
//                 const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
//                 const uint32_t a6 = right_bot;
//                 const uint32_t a7 = (right_bot << 1) | (col1_left_bot >> 31);
                
//                 if (row == 1) {
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
//                 }
                
//                 uint32_t bottom_xor, bottom_maj;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                
//                 uint32_t magic0, magic1, magic2, result;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_right_mid_xor), "r"(bottom_xor), "r"(center));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_right_mid_maj), "r"(c0_right_top_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
//                 output_val |= (uint64_t)result;
                
//                 c0_right_top_xor = c0_right_mid_xor;
//                 c0_right_mid_xor = bottom_xor;
//                 c0_right_top_maj = c0_right_mid_maj;
//                 c0_right_mid_maj = bottom_maj;
//             }
            
//             output[row-1][0] = output_val;
//         }
        
//         // ========== COLUMN 1 ==========
//         {
//             uint32_t left_top = (uint32_t)(storage[row-1][1] >> 32);
//             uint32_t right_top = (uint32_t)(storage[row-1][1]);
//             uint32_t left_mid = (uint32_t)(storage[row][1] >> 32);
//             uint32_t right_mid = (uint32_t)(storage[row][1]);
//             uint32_t left_bot = (uint32_t)(storage[row+1][1] >> 32);
//             uint32_t right_bot = (uint32_t)(storage[row+1][1]);
            
//             // For wrapping from column 0
//             uint32_t col0_right_top = (uint32_t)(storage[row-1][0]);
//             uint32_t col0_right_mid = (uint32_t)(storage[row][0]);
//             uint32_t col0_right_bot = (uint32_t)(storage[row+1][0]);
            
//             uint64_t output_val = 0;
            
//             // --- LEFT HALF (upper 32 bits) ---
//             {
//                 const uint32_t a0 = (col0_right_top << 1) | (left_top >> 31);  // Wrap from column 0
//                 const uint32_t a1 = left_top;
//                 const uint32_t a2 = (left_top << 1) | (right_top >> 31);
//                 const uint32_t a3 = (col0_right_mid << 1) | (left_mid >> 31);
//                 const uint32_t center = left_mid;
//                 const uint32_t a4 = (left_mid << 1) | (right_mid >> 31);
//                 const uint32_t a5 = (col0_right_bot << 1) | (left_bot >> 31);
//                 const uint32_t a6 = left_bot;
//                 const uint32_t a7 = (left_bot << 1) | (right_bot >> 31);
                
//                 if (row == 1) {
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
//                 }
                
//                 uint32_t bottom_xor, bottom_maj;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                
//                 uint32_t magic0, magic1, magic2, result;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_left_mid_xor), "r"(bottom_xor), "r"(center));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_left_mid_maj), "r"(c1_left_top_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
//                 output_val = ((uint64_t)result << 32);
                
//                 c1_left_top_xor = c1_left_mid_xor;
//                 c1_left_mid_xor = bottom_xor;
//                 c1_left_top_maj = c1_left_mid_maj;
//                 c1_left_mid_maj = bottom_maj;
//             }
            
//             // --- RIGHT HALF (lower 32 bits) ---
//             {
//                 const uint32_t a0 = (left_top << 31) | (right_top >> 1);
//                 const uint32_t a1 = right_top;
//                 const uint32_t a2 = right_top << 1;  // Wraps to column 0's left half
//                 const uint32_t a3 = (left_mid << 31) | (right_mid >> 1);
//                 const uint32_t center = right_mid;
//                 const uint32_t a4 = right_mid << 1;
//                 const uint32_t a5 = (left_bot << 31) | (right_bot >> 1);
//                 const uint32_t a6 = right_bot;
//                 const uint32_t a7 = right_bot << 1;
                
//                 if (row == 1) {
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
//                     asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
//                 }
                
//                 uint32_t bottom_xor, bottom_maj;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                
//                 uint32_t magic0, magic1, magic2, result;
//                 asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_right_mid_xor), "r"(bottom_xor), "r"(center));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_right_mid_maj), "r"(c1_right_top_maj));
//                 asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                
//                 output_val |= (uint64_t)result;
                
//                 c1_right_top_xor = c1_right_mid_xor;
//                 c1_right_mid_xor = bottom_xor;
//                 c1_right_top_maj = c1_right_mid_maj;
//                 c1_right_mid_maj = bottom_maj;
//             }
            
//             output[row-1][1] = output_val;
//         }
//     }
    
//     // Write outputs
//     out10 = output[0][0]; out11 = output[0][1];
//     out20 = output[1][0]; out21 = output[1][1];
//     out30 = output[2][0]; out31 = output[2][1];
//     out40 = output[3][0]; out41 = output[3][1];
// }

// __device__ inline void twoByFourCentalUpdate(uint64_t r00, uint64_t r01, uint64_t r10, uint64_t r11, uint64_t r20, uint64_t r21, uint64_t r30, uint64_t r31){
//     uint64_t out00, out01, out10, out11;
// }

__global__ void multistepKernel(uint64_t* globalData, int height, int width, int steps, int iterations) {
    // Shared memory for 16 warps, each warp has 240 uint64_t (80 rows × 3 columns)
    __shared__ uint64_t warpStorage[16][256];


//     // Start Y position of the tile including the 8-row halo above
//     int haloStartY = (globalStartY - 32 + height) & (height - 1);

//     // Load top halo  __shared__ uint64_t warpStorage[16][256];

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

    //(8 rows)
//     if (laneId < 4) {
//         for (int i = 0; i < 2; ++i) {
//             int localY = laneId * 2 + i; // rows 0–7
//             int globalY = (haloStartY + localY) & (height - 1);
//             warpStorage[warpId][localY * 3 + 0] = globalData[(globalY * width) + x_left];
//             warpStorage[warpId][localY * 3 + 1] = globalData[(globalY * width) + globalX];
//             warpStorage[warpId][localY * 3 + 2] = globalData[(globalY * width) + x_right];
//         }
//     }

//     // Load active rows (64 rows, 8..71)
//     int activeStart = 8 + laneId * 2;  // where this thread’s rows start
//     for (int i = 0; i < 2; ++i) {
//         int localY = activeStart + i;
//         int globalY = (haloStartY + localY) & (height - 1);
//         warpStorage[warpId][localY * 3 + 0] = globalData[(globalY * width) + x_left];
//         warpStorage[warpId][localY * 3 + 1] = globalData[(globalY * width) + globalX];
//         warpStorage[warpId][localY * 3 + 2] = globalData[(globalY * width) + x_right];
//     }

//     // Load bottom halo (8 rows)
//    if (laneId >= 28) {
//        for (int i = 0; i < 2; ++i) {
//            int localY = 72 + (laneId - 28) * 2 + i; // rows 72-79
//            int globalY = (haloStartY + localY) & (height - 1);
//             warpStorage[warpId][localY * 3 + 0] = globalData[(globalY * width) + x_left];
//             warpStorage[warpId][localY * 3 + 1] = globalData[(globalY * width) + globalX];
//             warpStorage[warpId][localY * 3 + 2] = globalData[(globalY * width) + x_right];
//         }
//     }


    //-------Before 8 iterations load register values within the thread (first 7 get to do 4 rows, the other 25 get to do 2 rows; as 78 rows to actually compute from)---------
    // if (laneId < 7){
    //     #pragma unroll
    //     for (int i = 0; i < 18; i++){
    //         localVals[i] = warpStorage[warpId][(laneId*12)+i];
    //     }

    //     for (int i = 0; i < steps; i++){
    //         uint64_t* r00, r01, r02, r10, r11, r12, r20, r21, r22, r30, r31, r32;
    //         RowDoubleLeftUpdate(r00,r10,);
    //         RowDoubleLeftUpdate(r01,r11,);
    //         RowDoubleLeftUpdate(r02,r12,);
    //         RowDoubleLeftUpdate(r20,r30,);
    //         RowDoubleLeftUpdate(r21,r31,);
    //         RowDoubleLeftUpdate(r22,r32,);

    //     }
    // }else{
    //     uint64_t localVals[12];

    //     #pragma unroll
    //     for (int i = 0; i < 12; i++){
    //         localVals[i] = warpStorage[warpId][84 + (laneId *6) + i];
    //     }

    //     for (int i = 0; i < steps; i++){
    //         uint64_t* r00, r01, r02, r10, r11, r12;
            
    //     }
    // }




    uint64_t* stuff;
    function(stuff, r00,r01)
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

    // if ((laneId >= 8) && (laneId < 24)){
    //     warpStorage[warpId][((laneId - 8) * 4)] = 
    //     warpStorage[warpId][((laneId - 8) * 4) + 1] = 
    //     warpStorage[warpId][((laneId - 8) * 4) + 2] = 
    //     warpStorage[warpId][((laneId - 8) * 4) + 3] = 
    // }

    // int localStartY = 8 + laneId * 2; 

    // for (int i = 0; i < 2; ++i) {
    //     int localY = localStartY + i;          
    //     int globalY = (haloStartY + localY) & (height - 1);  // Use haloStartY + localY
    //     globalData[globalY * width + globalX] = warpStorage[warpId][localY * 3 + 1];
    // }


}


void gol(uint64_t **current_ptr, uint64_t **next_ptr, int width, int height, int iterations) {
    uint64_t *current = *current_ptr;
    uint64_t *next = *next_ptr;
    size_t size = (width / 64) * height * sizeof(uint64_t);

    uint64_t *d_current;
    uint64_t *d_next;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);

    cudaMemcpy(d_current, *current_ptr, size, cudaMemcpyHostToDevice);

    dim3 blockDim(4, 128);
    dim3 gridDim(64,64);


    for (int iteration = 0; iteration < iterations; iteration++) {
        singleIteration<<<gridDim, blockDim>>>(d_current, d_next, height, arrayWidth);
        cudaDeviceSynchronize();

        uint64_t *tmp = d_current;
        d_current = d_next;
        d_next = tmp;
    }
    cudaMemcpy(*current_ptr, d_current, size, cudaMemcpyDeviceToHost);

    cudaFree(d_current);
    cudaFree(d_next);
}