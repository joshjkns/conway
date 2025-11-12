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

__device__ inline void RowDoubleLeftUpdate(uint64_t *nextTop, uint64_t *nextBottom, uint64_t topLeft,uint64_t top,uint64_t topRight,uint64_t middle11, uint64_t middle12,uint64_t middle13,uint64_t middle 21, uint64_t middle22,uint64_t middle23,uint64_t bottomLeft,uint64_t bottom,uint64_t bottomRight) {
    uint64_t tl = topLeft << 1;
    uint64_t t = topLeft;
    uint64_t tr = rotate_right(topLeft, top);
    uint64_t m11 = middle11 << 1;
    uint64_t m12 = middle11;
    uint64_t m13 = rotate_right(middle11, middle12);
    uint64_t m21 = middle21 << 1;
    uint64_t m22 = middle21;
    uint64_t m23 = rotate_right(middle21, middle22);
    uint64_t bl = bottomLeft << 1;
    uint64_t b = bottomLeft;
    uint64_t br = rotate_right(bottomLeft, bottom);
    
    
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
    *nextTop = (m11 & m12_count2) | m12_count3;
    *nextBottom = (m21 & m22_count2) | m22_count3;
}

__device__ inline void RowDoubleRightUpdate(uint64_t *nextTop, uint64_t *nextBottom, uint64_t topLeft,uint64_t top,uint64_t topRight,uint64_t middle11, uint64_t middle12,uint64_t middle13,uint64_t middle 21, uint64_t middle22,uint64_t middle23,uint64_t bottomLeft,uint64_t bottom,uint64_t bottomRight) {
    uint64_t tl = rotate_left(topRight, top);
    uint64_t t = topRight;
    uint64_t tr = topRight >> 1;
    uint64_t m11 = rotate_left(middle13, middle12);
    uint64_t m12 = middle13;
    uint64_t m13 = middle13 >> 1;
    uint64_t m21 = rotate_left(middle23, middle22);
    uint64_t m22 = middle23;
    uint64_t m23 = middle23 >> 1;
    uint64_t bl = rotate_left(bottomRight, bottom);
    uint64_t b = bottomRight;
    uint64_t br = bottomRight >> 1;
    
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
    *nextTop = (m13 & m12_count2) | m12_count3;
    *nextBottom = (m23 & m22_count2) | m22_count3;
}

__device__ inline void RowDoubleCentralUpdate(uint64_t *nextTop, uint64_t *nextBottom, uint64_t topLeft,uint64_t top,uint64_t topRight,uint64_t middle11, uint64_t middle12,uint64_t middle13,uint64_t middle 21, uint64_t middle22,uint64_t middle23,uint64_t bottomLeft,uint64_t bottom,uint64_t bottomRight) {
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
    *nextTop = (m12 & m12_count2) | m12_count3;
    *nextBottom = (m22 & m22_count2) | m22_count3;
}

__global__ void multistepKernel(uint64_t* globalData, int height, int width, int steps, int iterations) {
    // Shared memory for 16 warps, each warp has 240 uint64_t (80 rows × 3 columns)
   


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
        uint64_t r00 = warpStorage[warpId][(laneId * 8)]
        uint64_t r01 = warpStorage[warpId][(laneId * 8) + 1]
        uint64_t r10 = warpStorage[warpId][(laneId * 8) + 2]
        uint64_t r11 = warpStorage[warpId][(laneId * 8) + 3]
        uint64_t r20 = warpStorage[warpId][(laneId * 8) + 4]
        uint64_t r21 = warpStorage[warpId][(laneId * 8) + 5]
        uint64_t r30 = warpStorage[warpId][(laneId * 8) + 6]
        uint64_t r31 = warpStorage[warpId][(laneId * 8) + 7]
        uint64_t r40 = warpStorage[warpId][(laneId * 8) + 4]
        uint64_t r41 = warpStorage[warpId][(laneId * 8) + 5]
        uint64_t r50 = warpStorage[warpId][(laneId * 8) + 6]
        uint64_t r51 = warpStorage[warpId][(laneId * 8) + 7]


    }else if (laneId == 31){
        uint64_t r00 = warpStorage[warpId][(laneId * 8)]
        uint64_t r01 = warpStorage[warpId][(laneId * 8) + 1]
        uint64_t r10 = warpStorage[warpId][(laneId * 8) + 2]
        uint64_t r11 = warpStorage[warpId][(laneId * 8) + 3]
        uint64_t r20 = warpStorage[warpId][(laneId * 8) + 4]
        uint64_t r21 = warpStorage[warpId][(laneId * 8) + 5]
        uint64_t r30 = warpStorage[warpId][(laneId * 8) + 6]
        uint64_t r31 = warpStorage[warpId][(laneId * 8) + 7]


    }else{
        uint64_t r00 = warpStorage[warpId][(laneId * 8)]
        uint64_t r01 = warpStorage[warpId][(laneId * 8) + 1]
        uint64_t r10 = warpStorage[warpId][(laneId * 8) + 2]
        uint64_t r11 = warpStorage[warpId][(laneId * 8) + 3]
        uint64_t r20 = warpStorage[warpId][(laneId * 8) + 4]
        uint64_t r21 = warpStorage[warpId][(laneId * 8) + 5]
        uint64_t r30 = warpStorage[warpId][(laneId * 8) + 6]
        uint64_t r31 = warpStorage[warpId][(laneId * 8) + 7]
        uint64_t r40 = warpStorage[warpId][(laneId * 8) + 4]
        uint64_t r41 = warpStorage[warpId][(laneId * 8) + 5]
        uint64_t r50 = warpStorage[warpId][(laneId * 8) + 6]
        uint64_t r51 = warpStorage[warpId][(laneId * 8) + 7]

        
    }

    if ((laneId >= 8) && (laneId < 24)){
        warpStorage[warpId][((laneId - 8) * 4)] = 
        warpStorage[warpId][((laneId - 8) * 4) + 1] = 
        warpStorage[warpId][((laneId - 8) * 4) + 2] = 
        warpStorage[warpId][((laneId - 8) * 4) + 3] = 
    }

    //------------Do computation and write back to shared memory----------------



    //--------------------------------------------------------------------------
    int localStartY = 8 + laneId * 2; 

    for (int i = 0; i < 2; ++i) {
        int localY = localStartY + i;          
        int globalY = (haloStartY + localY) & (height - 1);  // Use haloStartY + localY
        globalData[globalY * width + globalX] = warpStorage[warpId][localY * 3 + 1];
    }

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