#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>
#include <stdint.h>

static inline uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint8_t truth) {
    uint32_t res = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t bit_a = (a >> i) & 1;
        uint32_t bit_b = (b >> i) & 1;
        uint32_t bit_c = (c >> i) & 1;
        uint8_t idx = (bit_c << 2) | (bit_b << 1) | bit_a;
        uint32_t truth_bit = (truth >> idx) & 1;
        // if (i >= 10 && i <= 14 && truth == 0b11101000) {
        //     printf("bit %d: a=%u b=%u c=%u idx=%u truth_bit=%u\n", 
        //            i, bit_a, bit_b, bit_c, idx, truth_bit);
        // }
        res |= truth_bit << i;
    }
    return res;
}

void print_binary64(uint64_t n) {
    for (int i = 63; i >= 0; i--) {
        uint64_t bit = (n >> i) & 1;
        printf("%" PRIu64, bit);
    }
    printf("\n");
}

void print_binary32(uint32_t n) {
    for (int i = 31; i >= 0; i--) {
        uint32_t bit = (n >> i) & 1;
        printf("%" PRIu32, bit);
    }
    printf("\n");
}

// Funnel shift left: concatenate high:low as 64-bit, shift left by n, return lower 32 bits
static inline uint32_t funnelshift_l(uint32_t high, uint32_t low, unsigned int n) {
    uint64_t combined = ((uint64_t)high << 32) | low;
    return (uint32_t)((combined << n) >> 32);
}

// Funnel shift right: concatenate high:low as 64-bit, shift right by n, return lower 32 bits
static inline uint32_t funnelshift_r(uint32_t high, uint32_t low, unsigned int n) {
    uint64_t combined = ((uint64_t)high << 32) | low;
    return (uint32_t)(combined >> n);
}

int main(){
    uint64_t storage[6][2];
    uint64_t r00 = 0x0000000000000000ULL;  // Row 0, left half
    uint64_t r01 = 0x0000000000000000ULL;  // Row 0, right half

    uint64_t r10 = 0x0000000000000000ULL;  // Row 1, left half
    uint64_t r11 = 0x0000000000000000ULL;  // Row 1, right half

    uint64_t r20 = 0x0000000000000000ULL;  // Row 2, left half: bits 1,2,3 set
    uint64_t r21 = 0x0000000000000E00ULL;  // Row 2, right half

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
                const uint32_t a2 = funnelshift_l(right_top, left_top, 1);
                const uint32_t a3 = left_mid >> 1;
                const uint32_t center = left_mid;
                const uint32_t a4 = funnelshift_l(right_mid, left_mid, 1);
                const uint32_t a5 = left_bot >> 1;
                const uint32_t a6 = left_bot;
                const uint32_t a7 = funnelshift_l(right_bot, left_bot, 1);
                
                uint32_t a8, a9, aA, b0 ,b1, b2, b3, c0, parity, two_three, result;

                a8 = lop3(a2, a1, a0, 0b10010110);
                b0 = lop3(a2, a1, a0, 0b11101000);
                a9 = lop3(a5, a4, a3, 0b10010110);
                b1 = lop3(a5, a4, a3, 0b11101000);
                aA = lop3(a8, a7, a6, 0b10010110);
                b2 = lop3(a8, a7, a6, 0b11101000);

                b3 = lop3(b2, b1, b0, 0b10010110);
                c0 = lop3(b2, b1, b0, 0b11101000);

                parity = lop3(center, aA, a9, 0b11110110);
                two_three = lop3(b3, aA, a9, 0b01111000);
                result = lop3(parity, two_three, c0, 0b01000000);


                //printf("row: %d, cells: %u0", row, result);

                output_val = ((uint64_t)result << 32);

            }
            
            // --- RIGHT HALF (lower / right 32 bits) ---
            {
                const uint32_t a0 = funnelshift_r(right_top, left_top, 1);
                const uint32_t a1 = right_top;
                const uint32_t a2 = funnelshift_l(col1_left_top, right_top, 1);  // Wrap to column 1
                const uint32_t a3 = funnelshift_r(right_mid, left_mid, 1);
                const uint32_t center = right_mid;
                const uint32_t a4 = funnelshift_l(col1_left_mid, right_mid, 1);
                const uint32_t a5 = funnelshift_r(right_bot, left_bot, 1);
                const uint32_t a6 = right_bot;
                const uint32_t a7 = funnelshift_l(col1_left_bot, right_bot, 1);
                
                //printf("\n%u %u %u %u %u %u %u %u\n", a0, a1, a2, a3, center, a4, a5, a6);

                uint32_t a8, a9, aA, b0 ,b1, b2, b3, c0, parity, two_three, result;

                a8 = lop3(a2, a1, a0, 0b10010110);
                b0 = lop3(a2, a1, a0, 0b11101000);
                a9 = lop3(a5, a4, a3, 0b10010110);
                b1 = lop3(a5, a4, a3, 0b11101000);
                aA = lop3(a8, a7, a6, 0b10010110);
                b2 = lop3(a8, a7, a6, 0b11101000);

                b3 = lop3(b2, b1, b0, 0b10010110);
                c0 = lop3(b2, b1, b0, 0b11101000);

                parity = lop3(center, aA, a9, 0b11110110);
                two_three = lop3(b3, aA, a9, 0b01111000);
                result = lop3(parity, two_three, c0, 0b01000000);

                output_val |= (uint64_t)result; 

            }
            
            output[row-1][0] = output_val;
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
                const uint32_t a0 = funnelshift_r(left_top,col0_right_top,1);  // Wrap from column 0
                const uint32_t a1 = left_top;
                const uint32_t a2 = funnelshift_l(left_top,right_top,1);
                const uint32_t a3 = funnelshift_r(left_mid,col0_right_mid,1);
                const uint32_t center = left_mid;
                const uint32_t a4 = funnelshift_l(left_mid,right_mid,1);
                const uint32_t a5 = funnelshift_r(left_bot,col0_right_bot,1);
                const uint32_t a6 = left_bot;
                const uint32_t a7 = funnelshift_l(left_bot,right_bot,1);
                
                uint32_t a8, a9, aA, b0 ,b1, b2, b3, c0, parity, two_three, result;

                a8 = lop3(a2, a1, a0, 0b10010110);
                b0 = lop3(a2, a1, a0, 0b11101000);
                a9 = lop3(a5, a4, a3, 0b10010110);
                b1 = lop3(a5, a4, a3, 0b11101000);
                aA = lop3(a8, a7, a6, 0b10010110);
                b2 = lop3(a8, a7, a6, 0b11101000);

                b3 = lop3(b2, b1, b0, 0b10010110);
                c0 = lop3(b2, b1, b0, 0b11101000);

                parity = lop3(center, aA, a9, 0b11110110);
                two_three = lop3(b3, aA, a9, 0b01111000);
                result = lop3(parity, two_three, c0, 0b01000000);

                //printf("row: %d, cells: %u0", row, result);

                output_val = ((uint64_t)result << 32);
            }
            
            // --- RIGHT HALF (lower / right 32 bits) ---
            {
                const uint32_t a0 = funnelshift_r(right_top, left_top, 1);
                const uint32_t a1 = right_top;
                const uint32_t a2 = right_top << 1;  // Wraps to column 0's left half
                const uint32_t a3 = funnelshift_r(right_mid, left_mid, 1);
                const uint32_t center = right_mid;
                const uint32_t a4 = right_mid << 1;
                const uint32_t a5 = funnelshift_r(right_bot, left_bot, 1);
                const uint32_t a6 = right_bot;
                const uint32_t a7 = right_bot << 1;
                
                uint32_t a8, a9, aA, b0 ,b1, b2, b3, c0, parity, two_three, result;

                printf("\n");
                print_binary32(a0);
                print_binary32(a1);
                print_binary32(a2);
                print_binary32(a3);
                print_binary32(center);
                print_binary32(a4);
                print_binary32(a5);
                print_binary32(a6);
                print_binary32(a7);

                a8 = lop3(a2, a1, a0, 0b10010110);
                printf("hfvgevf\n");
                print_binary32(a8);
                printf("hfvgevf\n");

                // ADD DEBUG HERE ↓
                printf("Calling lop3(a2=%u, a1=%u, a0=%u, 0b11101000)\n", a2, a1, a0);

                // Also test bit 12 manually BEFORE the call:
                uint32_t bit12_a = (a2 >> 12) & 1;
                uint32_t bit12_b = (a1 >> 12) & 1;
                uint32_t bit12_c = (a0 >> 12) & 1;
                uint8_t idx12 = (bit12_c << 2) | (bit12_b << 1) | bit12_a;
                uint32_t truth_bit12 = (0b11101000 >> idx12) & 1;
                printf("Bit 12 BEFORE call: a=%u b=%u c=%u idx=%u truth_bit=%u\n", 
                    bit12_a, bit12_b, bit12_c, idx12, truth_bit12);

                b0 = lop3(a2, a1, a0, 0b11101000);

                printf("Result b0 = %u\n", b0);
                printf("hfvgevf\n");
                print_binary32(b0);
                printf("hfvgevf\n");
                a9 = lop3(a5, a4, a3, 0b10010110);
                b1 = lop3(a5, a4, a3, 0b11101000);
                aA = lop3(a8, a7, a6, 0b10010110);
                b2 = lop3(a8, a7, a6, 0b11101000);

                b3 = lop3(b2, b1, b0, 0b10010110);
                c0 = lop3(b2, b1, b0, 0b11101000);

                parity = lop3(center, aA, a9, 0b11110110);
                two_three = lop3(b3, aA, a9, 0b01111000);
                result = lop3(parity, two_three, c0, 0b01000000);

                //printf("       row: %d, cells: %u0\n", row, result);

                printf("row: %d\n", row);
                print_binary32(result);

                output_val |= (uint64_t)result;
            }
            
            output[row-1][1] = output_val;
        }
    }
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 2; col++) {
            printf("output[%d][%d] =", row, col);
            print_binary64(output[row][col]);
            printf("\n");
        }
    }
    return 0;
}