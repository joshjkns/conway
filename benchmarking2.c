#include <stdint.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>
#include <stdbool.h>

// Naive byte-based GOL (for comparison and correctness reference)
void gol_naive(uint8_t **current_ptr, uint8_t **next_ptr, int width, int height, int iterations) {
    uint8_t *current = *current_ptr;
    uint8_t *next = *next_ptr;
    
    for (int iteration = 0; iteration < iterations; iteration++) {
        for (int y = 0; y < height; y++) {
            int yUp = (y - 1 + height) % height;
            int yDown = (y + 1) % height;
            
            for (int x = 0; x < width; x++) {
                int xLeft = (x - 1 + width) % width;
                int xRight = (x + 1) % width;
                
                // Count 8 neighbors
                int count = 
                    current[yUp * width + xLeft] +
                    current[yUp * width + x] +
                    current[yUp * width + xRight] +
                    current[y * width + xLeft] +
                    current[y * width + xRight] +
                    current[yDown * width + xLeft] +
                    current[yDown * width + x] +
                    current[yDown * width + xRight];
                
                // Apply rules
                int cell = current[y * width + x];
                if (cell) {
                    next[y * width + x] = (count == 2 || count == 3) ? 1 : 0;
                } else {
                    next[y * width + x] = (count == 3) ? 1 : 0;
                }
            }
        }
        
        uint8_t *tmp = current;
        current = next;
        next = tmp;
    }
    
    *current_ptr = current;
    *next_ptr = next;
}

void print_binary(uint64_t value) {
    for (int i = 63; i >= 0; i--) {
        putchar((value >> i) & 1 ? '1' : '0');
    }
    putchar('\n');
}


// Bit-parallel GOL implementation
static inline uint64_t rotate_left(uint64_t centre, uint64_t left){
    return (centre << 1) | (left >> 63);
}

static inline uint64_t rotate_right(uint64_t centre, uint64_t right){
    return (centre >> 1) | (right << 63);
}

void gol(uint64_t **current_ptr, uint64_t **next_ptr, int width, int height, int iterations) {
    uint64_t *current = *current_ptr;
    uint64_t *next = *next_ptr;
    for (int iteration = 0; iteration < iterations; iteration++) {
        int xElements = width / 64;
        //#pragma omp parallel for schedule(runtime)
        for(int y = 0; y < height; y++){
            int yUp = (y - 1 + height) % height;
            int yDown = (y + 1) % height;
            //printf("%d,%d\n", yUp, yDown);
            for (int x = 0; x < xElements; x++){
                int x_left = (x - 1 + xElements) % xElements;
                int x_right = (x + 1) % xElements;
                //printf("%d,%d\n", x_left, x_right);

                uint64_t topLeft = current[yUp * xElements + x_left];
                uint64_t top = current[yUp * xElements + x];
                uint64_t topRight = current[yUp * xElements + x_right];
                uint64_t left = current[y * xElements + x_left];
                uint64_t center = current[y * xElements + x];
                uint64_t right = current[y * xElements + x_right];
                uint64_t bottomLeft = current[yDown * xElements + x_left];
                uint64_t bottom = current[yDown * xElements + x];
                uint64_t bottomRight = current[yDown * xElements + x_right];
                printf("\n\n\n row:%d\n", y);
                uint64_t tl = rotate_left(top, topLeft);
                print_binary(tl);
                uint64_t t = top;
                print_binary(t);
                uint64_t tr = rotate_right(top, topRight);
                print_binary(tr);
                uint64_t l = rotate_left(center, left);
                print_binary(l);
                uint64_t r = rotate_right(center, right);
                print_binary(r);
                uint64_t bl = rotate_left(bottom, bottomLeft);
                print_binary(bl);
                uint64_t b = bottom;
                print_binary(b);
                uint64_t br = rotate_right(bottom, bottomRight);
                print_binary(br);
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
                // printf("%d,%d\n", yUp, yDown);
                // printf("%d,%d\n", x_left, x_right);
                printf("\nsum:%d\n", y);
                print_binary(tempBit2);
                print_binary(tempBit1);
                print_binary(tempBit0);

                //above just sums the number of ones, and represnts it in three 64 bit numbers, one bit in each number
                uint64_t finalTemp = (~tempBit2) & tempBit1;
                uint64_t two = (~tempBit0) & finalTemp;
                uint64_t three = tempBit0 & finalTemp;
                print_binary(two);
                print_binary(three);
                printf("center:\n");
                print_binary(center);
                next[y * xElements + x] = three | (two & center);
            }
        }
        printf("\nFinal result:\n");
        for (int x = 0; x < height; x++){
            print_binary(next[x]);
        }
        uint64_t *tmp = current;
        current = next;
        next = tmp;
        printf("\nyooloo\n");
    }
    *current_ptr = current;
    *next_ptr = next;
}

void print_row(uint64_t row) {
    for (int i = 63; i >= 0; i--) {
        putchar((row >> i) & 1 ? '#' : '.');
    }
    putchar('\n');
}

// ----------------- TEST HARNESS -----------------
void print_grid_uint8(int width, int height, uint8_t *grid) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%c", grid[y * width + x] ? '#' : '.');
        }
        printf("\n");
    }
}

void print_grid_uint64(int width, int height, uint64_t *grid) {
    int xElements = width / 64;
    for (int y = 0; y < height; y++) {
        for (int xe = 0; xe < xElements; xe++) {
            for (int bit = 63; bit >= 0; bit--) {
                printf("%c", (grid[y * xElements + xe] & ((uint64_t)1 << bit)) ? '#' : '.');
            }
        }
        printf("\n");
    }
}

// Convert byte-based grid to 64-bit word grid
void bytes_to_uint64(uint8_t *src, uint64_t *dst, int width, int height) {
    int xElements = width / 64;
    for (int y = 0; y < height; y++) {
        for (int xe = 0; xe < xElements; xe++) {
            uint64_t word = 0;
            for (int bit = 0; bit < 64; bit++) {
                int x = xe * 64 + bit;
                word <<= 1;
                if (x < width && src[y * width + x]) word |= 1;
            }
            dst[y * xElements + xe] = word;
        }
    }
}

// Convert 64-bit word grid back to byte grid
void uint64_to_bytes(uint64_t *src, uint8_t *dst, int width, int height) {
    int xElements = width / 64;
    for (int y = 0; y < height; y++) {
        for (int xe = 0; xe < xElements; xe++) {
            uint64_t word = src[y * xElements + xe];
            for (int bit = 63; bit >= 0; bit--) {
                int x = xe * 64 + (63 - bit);
                if (x < width)
                    dst[y * width + x] = (word & ((uint64_t)1 << bit)) ? 1 : 0;
            }
        }
    }
}

void test_gol_correctness() {
    int width = 64;  // must be multiple of 64
    int height = 8;
    int iterations = 1;

    // Allocate grids
    uint8_t *grid_bytes = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    uint8_t *naive_result = (uint8_t*)calloc(width * height, sizeof(uint8_t));
    uint8_t *cuda_result_bytes = (uint8_t*)calloc(width * height, sizeof(uint8_t));

    int xElements = width / 64;
    uint64_t *grid64 = (uint64_t*)calloc(height * xElements, sizeof(uint64_t));
    uint64_t *cuda_result64 = (uint64_t*)calloc(height * xElements, sizeof(uint64_t));

    // Initialize a test pattern (glider in top-left corner)
    grid_bytes[1*width + 2] = 1;
    grid_bytes[2*width + 3] = 1;
    grid_bytes[3*width + 1] = 1;
    grid_bytes[3*width + 2] = 1;
    grid_bytes[3*width + 3] = 1;

    printf("Initial grid:\n");
    print_grid_uint8(width, height, grid_bytes);

    // Copy initial state to naive result and 64-bit grid
    memcpy(naive_result, grid_bytes, width * height);
    bytes_to_uint64(grid_bytes, grid64, width, height);
    memcpy(cuda_result64, grid64, sizeof(uint64_t) * height * xElements);

    // Run naive GOL
    gol_naive(&naive_result, &naive_result, width, height, iterations);

    // Run bit-parallel GOL
    gol(&grid64, &cuda_result64, width, height, iterations);

    uint64_to_bytes(grid64, cuda_result_bytes, width, height);

    printf("\nNaive result:\n");
    print_grid_uint8(width, height, naive_result);

    printf("\nBit-parallel result:\n");
    print_grid_uint8(width, height, cuda_result_bytes);

    // Compare results
    bool ok = true;
    for (int i = 0; i < width * height; i++) {
        if (naive_result[i] != cuda_result_bytes[i]) {
            int y = i / width;
            int x = i % width;
            printf("Mismatch at (%d,%d): naive=%d, parallel=%d\n",
                   y, x, naive_result[i], cuda_result_bytes[i]);
            ok = false;
        }
    }

    if(ok) {
        printf("\nTest PASSED ✅\n");
    } else {
        printf("\nTest FAILED ❌\n");
    }

    free(grid_bytes);
    free(naive_result);
    free(cuda_result_bytes);
    free(grid64);
    free(cuda_result64);
}

int main() {
    test_gol_correctness();
    return 0;
}
