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
        res |= ((truth >> idx) & 1) << i;
    }
    return res;
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

static uint32_t sub_step(const unsigned int a0,
                                const unsigned int a1,
                                const unsigned int a2,
                                const unsigned int a3,
                                const unsigned int a4,
                                const unsigned int a5,
                                const unsigned int a6,
                                const unsigned int a7,
                                const unsigned int top_xor,
                                const unsigned int bottom_xor,
                                const unsigned int top_maj,
                                const unsigned int bottom_maj,
                                unsigned int center) {
    unsigned int a8, a9, aA, b0, b1, b2, magic0, magic1, magic2;

    // stage 1
    //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(aA) : "r"(top_xor), "r"(a4), "r"(a3));
    aA = lop3(top_xor, a4, a3, 0b10010110);
    //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(b2) : "r"(top_xor), "r"(a4), "r"(a3));
    b2 = lop3(top_xor, a4, a3, 0b11101000);

    // magic stage dreamt up by an insane SAT-solver
    //asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(bottom_xor), "r"(aA), "r"(center));
    magic0 = lop3(bottom_xor, aA, center, 0b00111110);
    //asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(b2));
    magic1 = lop3(magic0, center, b2, 0b01011011);
    //asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(bottom_maj), "r"(top_maj));
    magic2 = lop3(magic1, bottom_maj, top_maj, 0b10010001);
    //asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(center) : "r"(magic2), "r"(magic0), "r"(magic1));
    center = lop3(magic2, magic0, magic1, 0b01011000);

    return center;
}

int main(){
    uint64_t storage[6][2];
    uint64_t r00 = 0x0000000000000000ULL;  // Row 0, left half
    uint64_t r01 = 0x0000000000000000ULL;  // Row 0, right half

    uint64_t r10 = 0x0000000000000000ULL;  // Row 1, left half
    uint64_t r11 = 0x0000000000000000ULL;  // Row 1, right half

    uint64_t r20 = 0x000000000000000EULL;  // Row 2, left half: bits 1,2,3 set
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
    
    // Reuse chains for column 0
    uint32_t c0_left_top_xor, c0_left_mid_xor, c0_left_top_maj, c0_left_mid_maj;
    uint32_t c0_right_top_xor, c0_right_mid_xor, c0_right_top_maj, c0_right_mid_maj;
    
    // Reuse chains for column 1
    uint32_t c1_left_top_xor, c1_left_mid_xor, c1_left_top_maj, c1_left_mid_maj;
    uint32_t c1_right_top_xor, c1_right_mid_xor, c1_right_top_maj, c1_right_mid_maj;
    
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
                
                uint32_t a8, a9, aA, b0 ,b1, b2, magic0, magic1, magic2, result;

                a8 = lop3(a2, a1, a0, 0b10010110);
                b0 = lop3(a2, a1, a0, 0b11101000);
                a9 = lop3(a4, a3, center, 0b10010110);
                b1 = lop3(a4, a3, center, 0b11101000);
                aA = lop3(a8, a6, a5, 0b10010110);
                b2 = lop3(a8, a6, a5, 0b11101000);
                magic0 = lop3(a9, aA, center, 0b00111110);
                magic1 = lop3(magic0, center, b2, 0b01011011);
                magic2 = lop3(magic1, b1, b0, 0b10010001);
                result = lop3(magic2, magic0, magic1, 0b01011000);

                output_val = ((uint64_t)result << 32);

                // if (row == 1) {
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                //     c0_left_top_xor = lop3(a2, a1, a0, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                //     c0_left_top_maj = lop3(a2, a1, a0, 0b11101000);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                //     c0_left_mid_xor = lop3(a4, a3, center, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                //     c0_left_mid_maj = lop3(a4, a3, center, 0b11101000);
                // }
                
                // uint32_t c0_left_bottom_xor, c0_left_bottom_maj;
                // //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                // c0_left_bottom_xor = lop3(a7, a6, a5, 0b10010110);
                // //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                // c0_left_bottom_maj = lop3(a7, a6, a5, 0b11101000);
                
                // // // Magic stage
                // // uint32_t magic0, magic1, magic2, result;
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_left_mid_xor), "r"(bottom_xor), "r"(center));
                // // magic0 = lop3(c0_left_mid_xor, bottom_xor, center, 0b00111110);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                // // magic1 = lop3(magic0, center, bottom_maj, 0b01011011);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_left_mid_maj), "r"(c0_left_top_maj));
                // // magic2 = lop3(magic1, c0_left_mid_maj, c0_left_top_maj, 0b10010001);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                // // result = lop3(magic2, magic0, magic1, 0b01011000);
                
                // uint32_t result;
                // result = sub_step(a0,a1,a2,a3,a4,a5,a6,a7,c0_left_top_xor, c0_left_bottom_xor,c0_left_top_maj,c0_left_bottom_maj,center);
                
                // c0_left_top_xor = c0_left_mid_xor;
                // c0_left_mid_xor = c0_left_bottom_xor;
                // c0_left_top_maj = c0_left_mid_maj;
                // c0_left_mid_maj = c0_left_bottom_maj;
            }
            
            // --- RIGHT HALF (lower / right 32 bits) ---
            {
                const uint32_t a0 = funnelshift_r(right_top, left_top, 1);
                const uint32_t a1 = right_top;
                const uint32_t a2 = funnelshift_l(right_top, col1_left_top, 1);  // Wrap to column 1
                const uint32_t a3 = funnelshift_r(right_mid, left_mid, 1);
                const uint32_t center = right_mid;
                const uint32_t a4 = funnelshift_l(right_mid, col1_left_mid, 1);
                const uint32_t a5 = funnelshift_r(right_bot, left_bot, 1);
                const uint32_t a6 = right_bot;
                const uint32_t a7 = funnelshift_l(right_bot, col1_left_bot, 1);
                
                uint32_t a8, a9, aA, b0 ,b1, b2, magic0, magic1, magic2, result;

                a8 = lop3(a2, a1, a0, 0b10010110);
                b0 = lop3(a2, a1, a0, 0b11101000);
                a9 = lop3(a4, a3, center, 0b10010110);
                b1 = lop3(a4, a3, center, 0b11101000);
                aA = lop3(a8, a6, a5, 0b10010110);
                b2 = lop3(a8, a6, a5, 0b11101000);
                magic0 = lop3(a9, aA, center, 0b00111110);
                magic1 = lop3(magic0, center, b2, 0b01011011);
                magic2 = lop3(magic1, b1, b0, 0b10010001);
                result = lop3(magic2, magic0, magic1, 0b01011000);

                output_val |= (uint64_t)result; 

                // if (row == 1) {
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                //     c0_right_top_xor = lop3(a2, a1, a0, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                //     c0_right_top_maj = lop3(a2, a1, a0, 0b11101000);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c0_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                //     c0_right_mid_xor = lop3(a4, a3, center, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c0_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                //     c0_right_mid_maj = lop3(a4, a3, center, 0b11101000);
                // }
                
                // uint32_t c0_right_bottom_xor, c0_right_bottom_maj;
                // //sm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                // c0_right_bottom_xor = lop3(a7, a6, a5, 0b10010110);
                // //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                // c0_right_bottom_maj = lop3(a7, a6, a5, 0b11101000);
                
                // // uint32_t magic0, magic1, magic2, result;
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c0_right_mid_xor), "r"(bottom_xor), "r"(center));
                // // magic0 = lop3(c0_right_mid_xor, c0_right_bottom_xor, center, 0b00111110);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                // // magic1 = lop3(magic0, center, c0_right_bottom_maj, 0b01011011);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c0_right_mid_maj), "r"(c0_right_top_maj));
                // // magic2 = lop3(magic1, c0_right_mid_maj, c0_right_top_maj, 0b10010001);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                // // result = lop3(magic2, magic0, magic1, 0b01011000);
                
                // uint32_t result;
                // result = sub_step(a0,a1,a2,a3,a4,a5,a6,a7,c0_right_top_xor, c0_right_bottom_xor,c0_right_top_maj,c0_right_bottom_maj,center);
  
                
                // c0_right_top_xor = c0_right_mid_xor;
                // c0_right_mid_xor = c0_right_bottom_xor;
                // c0_right_top_maj = c0_right_mid_maj;
                // c0_right_mid_maj = c0_right_bottom_maj;
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
                const uint32_t a0 = funnelshift_l(left_top,col0_right_top,1);  // Wrap from column 0
                const uint32_t a1 = left_top;
                const uint32_t a2 = funnelshift_l(right_top,left_top,1);
                const uint32_t a3 = funnelshift_l(left_mid,col0_right_mid,1);
                const uint32_t center = left_mid;
                const uint32_t a4 = funnelshift_l(right_mid,left_mid,1);
                const uint32_t a5 = funnelshift_l(left_bot,col0_right_bot,1);
                const uint32_t a6 = left_bot;
                const uint32_t a7 = funnelshift_l(right_bot,left_bot,1);
                
                uint32_t a8, a9, aA, b0 ,b1, b2, magic0, magic1, magic2, result;

                a8 = lop3(a2, a1, a0, 0b10010110);
                b0 = lop3(a2, a1, a0, 0b11101000);
                a9 = lop3(a4, a3, center, 0b10010110);
                b1 = lop3(a4, a3, center, 0b11101000);
                aA = lop3(a8, a6, a5, 0b10010110);
                b2 = lop3(a8, a6, a5, 0b11101000);
                magic0 = lop3(a9, aA, center, 0b00111110);
                magic1 = lop3(magic0, center, b2, 0b01011011);
                magic2 = lop3(magic1, b1, b0, 0b10010001);
                result = lop3(magic2, magic0, magic1, 0b01011000);

                output_val = ((uint64_t)result << 32);

                // if (row == 1) {
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                //     c1_left_top_xor = lop3(a2, a1, a0, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                //     c1_left_top_maj = lop3(a2, a1, a0, 0b11101000);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_left_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                //     c1_left_mid_xor = lop3(a4, a3, center, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_left_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                //     c1_left_mid_maj = lop3(a4, a3, center, 0b11101000);
                // }
                
                // uint32_t c1_left_bottom_xor, c1_left_bottom_maj;
                // //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                // c1_left_bottom_xor = lop3(a7, a6, a5, 0b10010110);
                // //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                // c1_left_bottom_maj = lop3(a7, a6, a5, 0b11101000);
                
                // // uint32_t magic0, magic1, magic2, result;
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_left_mid_xor), "r"(bottom_xor), "r"(center));
                // // magic0 = lop3(c1_left_mid_xor, bottom_xor, center, 0b00111110);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                // // magic1 = lop3(magic0, center, bottom_maj, 0b01011011);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_left_mid_maj), "r"(c1_left_top_maj));
                // // magic2 = lop3(magic1, c1_left_mid_maj, c1_left_top_maj, 0b10010001);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                // // result = lop3(magic2, magic0, magic1, 0b01011000);
                
                // uint32_t result;
                // result = sub_step(a0,a1,a2,a3,a4,a5,a6,a7,c1_left_top_xor, c1_left_bottom_xor,c1_left_top_maj,c1_left_bottom_maj,center);

                //output_val = ((uint64_t)result << 32);
                
                // c1_left_top_xor = c1_left_mid_xor;
                // c1_left_mid_xor = c1_left_bottom_xor;
                // c1_left_top_maj = c1_left_mid_maj;
                // c1_left_mid_maj = c1_left_bottom_maj;
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
                
                uint32_t a8, a9, aA, b0 ,b1, b2, magic0, magic1, magic2, result;

                a8 = lop3(a2, a1, a0, 0b10010110);
                b0 = lop3(a2, a1, a0, 0b11101000);
                a9 = lop3(a4, a3, center, 0b10010110);
                b1 = lop3(a4, a3, center, 0b11101000);
                aA = lop3(a8, a6, a5, 0b10010110);
                b2 = lop3(a8, a6, a5, 0b11101000);
                magic0 = lop3(a9, aA, center, 0b00111110);
                magic1 = lop3(magic0, center, b2, 0b01011011);
                magic2 = lop3(magic1, b1, b0, 0b10010001);
                result = lop3(magic2, magic0, magic1, 0b01011000);

                output_val |= (uint64_t)result;

                // if (row == 1) {
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_top_xor) : "r"(a2), "r"(a1), "r"(a0));
                //     c1_right_top_xor = lop3(a2, a1, a0, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_top_maj) : "r"(a2), "r"(a1), "r"(a0));
                //     c1_right_top_maj = lop3(a2, a1, a0, 0b11101000);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(c1_right_mid_xor) : "r"(a4), "r"(a3), "r"(center));
                //     c1_right_mid_xor = lop3(a4, a3, center, 0b10010110);
                //     //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(c1_right_mid_maj) : "r"(a4), "r"(a3), "r"(center));
                //     c1_right_mid_maj = lop3(a4, a3, center, 0b11101000);
                // }
                
                // uint32_t c1_right_bottom_xor, c1_right_bottom_maj;
                // //asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                // c1_right_bottom_xor = lop3(a7, a6, a5, 0b10010110);
                // //asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(bottom_maj) : "r"(a7), "r"(a6), "r"(a5));
                // c1_right_bottom_maj = lop3(a7, a6, a5, 0b11101000);
                
                // // uint32_t magic0, magic1, magic2, result;
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(c1_right_mid_xor), "r"(bottom_xor), "r"(center));
                // // magic0 = lop3(c1_right_mid_xor, bottom_xor, center, 0b00111110);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(center), "r"(bottom_maj));
                // // magic1 = lop3(magic0, center, bottom_maj, 0b01011011);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(c1_right_mid_maj), "r"(c1_right_top_maj));
                // // magic2 = lop3(magic1, c1_right_mid_maj, c1_right_top_maj, 0b10010001);
                // // //asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result) : "r"(magic2), "r"(magic0), "r"(magic1));
                // //result = lop3(magic2, magic0, magic1, 0b01011000);
                
                // uint32_t result;
                // result = sub_step(a0,a1,a2,a3,a4,a5,a6,a7,c1_right_top_xor, c1_right_bottom_xor,c1_right_top_maj,c1_right_bottom_maj,center);

                
                // c1_right_top_xor = c1_right_mid_xor;
                // c1_right_mid_xor = c1_right_bottom_xor;
                // c1_right_top_maj = c1_right_mid_maj;
                // c1_right_mid_maj = c1_right_bottom_maj;
            }
            
            output[row-1][1] = output_val;
        }
    }
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 2; col++) {
            printf("output[%d][%d] = %lu\n", row, col, output[row][col]);
        }
    }
    // r10 = output[1][0];
    // r11 = output[1][1];
    // r20 = output[2][0];
    // r21 = output[2][1];
    // r30 = output[3][0];
    // r31 = output[3][1];
    // r40 = output[4][0];
    // r41 = output[4][1];
    return 0;
}