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
    uint64_t left = current[y * width + x_left];
    uint64_t center = current[y * width + x];
    uint64_t right = current[y * width + x_right];
    uint64_t bottomLeft = current[y_down * width + x_left];
    uint64_t bottom = current[y_down * width + x];
    uint64_t bottomRight = current[y_down * width + x_right];

    uint64_t tl = rotate_left(top, topLeft);
    uint64_t t = top;
    uint64_t tr = rotate_right(top, topRight);
    uint64_t l = rotate_left(center, left);
    uint64_t r = rotate_right(center, right);
    uint64_t bl = rotate_left(bottom, bottomLeft);
    uint64_t b = bottom;
    uint64_t br = rotate_right(bottom, bottomRight);

    uint64_t sum1 = tl ^ t;
    uint64_t sum2 = tr ^ l;
    uint64_t sum3 = r ^ bl;
    uint64_t sum4 = b ^ br;

    uint64_t carry1 = tl & t;
    uint64_t carry2 = tr & l;
    uint64_t carry3 = r & bl;
    uint64_t carry4 = b & br;

    uint64_t sum12 = sum1 ^ sum2;
    uint64_t tempCarry1 = (sum1 & sum2) | ((carry1 ^ carry2) & sum12);
    uint64_t tempCarry2 = (carry1 ^ carry2);

    uint64_t sum34 = sum3 ^ sum4;
    uint64_t tempCarry3 = (sum3 & sum4) | ((carry3 ^ carry4) & sum34);
    uint64_t tempCarry4 = (carry3 ^ carry4);

    uint64_t bit0 = sum12 ^ sum34;
    uint64_t temp = sum12 & sum34;
    uint64_t bit1 = temp ^ (tempCarry1 ^ tempCarry3);
    temp = (temp & (tempCarry2 ^ tempCarry4)) | (tempCarry2 & tempCarry4);
    uint64_t bit2 = temp ^ (tempCarry1 ^ tempCarry3);

    uint64_t three = ~bit2 & bit1 & bit0;
    uint64_t two = ~bit2 & bit1 & ~bit0;
    next[y * width + x] = three | (two & center);
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
