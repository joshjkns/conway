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

__global__ void singleIteration(uint64_t *current, uint64_t *next, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y_up = (y - 1 + height) % height;
    int y_down = (y + 1) % height;
    int x_left = (x - 1 + width) % width;
    int x_right = (x + 1) % width;

    uint64_t topLeft = current[y_up * width + x_left];
    uint64_t top = current[y_up * width + x];
    uint64_t topRight = current[y_up * width + x_right];
    uint64_t middle11 = current[y * width + x_left];
    uint64_t middle12 = current[y * width + x];
    uint64_t middle13 = current[y * width + x_right];
    uint64_t middle21 = current[y * width + x_left];
    uint64_t middle22 = current[y * width + x];
    uint64_t middle23 = current[y * width + x_right];
    uint64_t bottomLeft = current[y_down * width + x_left];
    uint64_t bottom = current[y_down * width + x];
    uint64_t bottomRight = current[y_down * width + x_right];

    uint64_t tl = rotate_left(top, topLeft);
    uint64_t t = top;
    uint64_t tr = rotate_right(top, topRight);
    uint64_t m11 = rotate_left(middle12, left);
    uint64_t m12 = middle12;
    uint64_t m13 = rotate_right(middle12, right);
    uint64_t m21 = rotate_left(middle22, left);
    uint64_t m22 = middle22;
    uint64_t m23 = rotate_right(middle22, right);
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
    next[y1 * width + x1] = (m12 & m12_count2) | m12_count3;
    next[y2 * width + x2] = (m22 & m22_count2) | m22_count3;
}

void gol(uint64_t **current_ptr, uint64_t **next_ptr, int width, int height, int iterations) {
    int wordsWidth = (width + 63) / 64;
    uint64_t *current = *current_ptr;
    uint64_t *next = *next_ptr;
    size_t size = wordsWidth * height * sizeof(uint64_t);

    uint64_t *d_current;
    uint64_t *d_next;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);

    cudaMemcpy(d_current, *current_ptr, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((wordsWidth + blockDim.x - 1)/blockDim.x, (height + blockDim.y - 1)/blockDim.y);


    for (int iteration = 0; iteration < iterations; iteration++) {
        singleIteration<<<gridDim, blockDim>>>(d_current, d_next, height, wordsWidth);
        cudaDeviceSynchronize();

        uint64_t *tmp = d_current;
        d_current = d_next;
        d_next = tmp;
    }
    cudaMemcpy(*current_ptr, d_current, size, cudaMemcpyDeviceToHost);

    cudaFree(d_current);
    cudaFree(d_next);
}
