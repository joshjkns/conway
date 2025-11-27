#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

__device__ inline uint64_t rotate_left(uint64_t centre, uint64_t left) {
    return (centre << 1) | (left >> 63);
}

__device__ inline uint64_t rotate_right( uint64_t centre, uint64_t right) {
    return (centre >> 1) | (right << 63);
}


__device__ inline int constrainValue(int value, int constraint) {
    if (value < 0) {
        value += constraint;
    } else if (value >= constraint) {
        value -= constraint;
    }
    return value;
}

__global__ void singleIteration(uint64_t *current, uint64_t *next, int height, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;

    int xElements = width;

    int yUp = constrainValue((y-1+height), height);
    int yDown = constrainValue((y+1), height);
    int x_left = constrainValue((x-1+width), width);
    int x_right = constrainValue((x+1), width);

    uint64_t topLeft = current[yUp * xElements + x_left];
    uint64_t top = current[yUp * xElements + x];
    uint64_t topRight = current[yUp * xElements + x_right];
    uint64_t left = current[y * xElements + x_left];
    uint64_t center = current[y * xElements + x];
    uint64_t right = current[y * xElements + x_right];
    uint64_t bottomLeft = current[yDown * xElements + x_left];
    uint64_t bottom = current[yDown * xElements + x];
    uint64_t bottomRight = current[yDown * xElements + x_right];

    uint64_t tl = rotate_left(top, topLeft);
    uint64_t t = top;
    uint64_t tr = rotate_right(top, topRight);
    uint64_t l = rotate_left(center, left);
    uint64_t r = rotate_right(center, right);
    uint64_t bl = rotate_left(bottom, bottomLeft);
    uint64_t b = bottom;
    uint64_t br = rotate_right(bottom, bottomRight);

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

    next[y * xElements + x] = three | (two & center);
}

extern "C" void iterationsRun(uint64_t **current_ptr, uint64_t **next_ptr, int width, int height, int iterations) {
    int wordsWidth = (width + 63) / 64;
    size_t size = wordsWidth * height * sizeof(uint64_t);

    uint64_t *d_current, *d_next;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);
    cudaMemcpy(d_current, *current_ptr, size, cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8);
    dim3 gridDim((wordsWidth + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    for (int i = 0; i < iterations; i++) {
        singleIteration<<<gridDim, blockDim>>>(d_current, d_next, height, wordsWidth);
        cudaDeviceSynchronize();
        uint64_t *tmp = d_current; d_current = d_next; d_next = tmp;
    }

    cudaMemcpy(*current_ptr, d_current, size, cudaMemcpyDeviceToHost);
    cudaFree(d_current);
    cudaFree(d_next);
}

