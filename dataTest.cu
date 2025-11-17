#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <inttypes.h>
// Fixed constants
#define WIDTH 16384
#define HEIGHT 16384
#define ITERATIONS 100000
#define WORK_GROUP_SIZE 32
#define WORK_PER_THREAD 16
#define STEP_SIZE 16

// Calculated constants
#define SIMULATED_ROWS (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * STEP_SIZE)  // 480, 480 + 32 = 512 is teh rows a block can actually do, so -16 in top and bottom for padding
#define HORIZONTAL_GROUPS ((WIDTH + 31) / 32)  // 512
#define VERTICAL_GROUPS ((HEIGHT + SIMULATED_ROWS - 1) / SIMULATED_ROWS)  // 35
#define PADDED_COLUMNS (HORIZONTAL_GROUPS + 2)  // 514
#define PADDED_HEIGHT (VERTICAL_GROUPS * SIMULATED_ROWS + 2 * STEP_SIZE)  // 16832
#define BUFFER_SIZE (PADDED_COLUMNS * PADDED_HEIGHT)  // 8,651,648

#define CLIP_TOP_LY ((STEP_SIZE + WORK_PER_THREAD - 1) / WORK_PER_THREAD - 1)
#define CLIP_TOP_OFFSET (STEP_SIZE - CLIP_TOP_LY * WORK_PER_THREAD)
#define CLIP_BOTTOM_LY (WORK_GROUP_SIZE - 1 - CLIP_TOP_LY)
#define CLIP_BOTTOM_OFFSET (WORK_PER_THREAD + 1 - CLIP_TOP_OFFSET)

__device__ inline void load_uint4(uint32_t *x, uint32_t *y, uint32_t *z, uint32_t *w, const uint32_t *addr) {
    asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(*x), "=r"(*y), "=r"(*z), "=r"(*w) : "l"(addr));
}

__device__ inline void permute(uint32_t *dest, uint32_t left, uint32_t right) {
    asm("prmt.b32 %0, %1, %2, 0x1076;" : "=r"(*dest) : "r"(left), "r"(right));
}

__global__ void sharedLoad(const uint32_t *globalData) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int width = 256;
    int height = 16384;
    unsigned long long int totalTime = 0;

    // Shared memory for 16 warps, each warp has 256 uint64_t (128 rows × 2 columns)
    __shared__ uint64_t warpStorage[16][256];
    // uint64_t totalTime = 0;
    // Thread/block coordinates

    // Warp identification
    int warpId = (ty >> 5) + (tx * 4);  // 16 warps per block
    int laneId = ty & 31;  // 31 is 0x1F, which is 2^5 - 1 (binary: 11111)
    // int laneId = constrainValue(ty, 32);               // 0–31 within the warp

    // Global grid position
    int globalX = (blockIdx.x * blockDim.x) + tx;
    int globalXleft  = (globalX - 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)
    int globalXright = (globalX + 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)
    
    unsigned long long int t0 = clock64();

    float yQuarter = (ty / 32) * 0.25f;
    unsigned long long int t1 = clock64();
    totalTime += t1 - t0;

    int centralWarpStartY = (blockIdx.y * 256) + (256 * yQuarter);
    //printf("startY: %d",centralWarpStartY);
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
    //printf("heuhjbef");
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
}

__global__ void globalLoad(const uint32_t *globalData) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int width = 256;
    int height = 16384;
    unsigned long long int totalTime = 0;

    uint64_t storage[4][2];

    // Warp identification
    int warpId = (ty >> 5) + (tx * 4);  // 16 warps per block
    int laneId = ty & 31;  // 31 is 0x1F, which is 2^5 - 1 (binary: 11111)
    // int laneId = constrainValue(ty, 32);               // 0–31 within the warp

    // Global grid position
    int globalX = (blockIdx.x * blockDim.x) + tx;
    int globalXleft  = (globalX - 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)
    int globalXright = (globalX + 1) & (width - 1); // X neighbors (wrap-around since width is power of 2)
    
    unsigned long long int t0 = clock64();

    float yQuarter = (ty / 32) * 0.25f;
    unsigned long long int t1 = clock64();
    totalTime += t1 - t0;

    int centralWarpStartY = (blockIdx.y * 256) + (256 * yQuarter);
    //printf("startY: %d",centralWarpStartY);
    int centralWarpEndY = centralWarpStartY + 64;

    int haloStartY = centralWarpStartY - 32;
    int haloEndY = centralWarpEndY + 32;
    for (int i = 0; i < 4; i++){
        int localY = haloStartY + (laneId * 4) + i;
        int globalY = (localY) & (height - 1);
        uint64_t left = globalData[(globalY * width) + globalXleft];
        uint64_t middle = globalData[(globalY * width) + globalX];
        uint64_t right = globalData[(globalY * width) + globalXright];
        storage[i][0] = (left << 32) | (middle >> 32);  
        storage[i][1] = (middle << 32) | (right >> 32); 
    }
    __syncthreads();
    //printf("heuhjbef");
    uint64_t r00;
    uint64_t r01;
    uint64_t r50;
    uint64_t r51;
    int blockStartY = (haloStartY + (laneId * 4) - 1) & (height - 1);
    int blockEndY = (haloStartY + (laneId * 4) + 4) & (height - 1);
    uint64_t r00 = (globalData[(blockStartY * width) + globalXleft] << 32) | (globalData[(blockStartY * width) + globalX] >> 32);
    uint64_t r01 = (globalData[(blockStartY * width) + globalX] << 32) | (globalData[(blockStartY * width) + globalXright] >> 32);
    uint64_t r10 = storage[0][0];
    uint64_t r11 = storage[0][1];
    uint64_t r20 = storage[1][0];
    uint64_t r21 = storage[1][1];
    uint64_t r30 = storage[2][0];
    uint64_t r31 = storage[2][1];
    uint64_t r40 = storage[3][0];
    uint64_t r41 = storage[3][1];
    uint64_t r50 = (globalData[(blockEndY * width) + globalXleft] << 32) | (globalData[(blockEndY * width) + globalX] >> 32);
    uint64_t r51 = (globalData[(blockEndY * width) + globalX] << 32) | (globalData[(blockEndY * width) + globalXright] >> 32);
}

void set_cell(uint32_t *buffer, int x, int y) {
    int column = x / 32 + 1;
    uint32_t bit_mask = 0x80000000 >> (x % 32);
    size_t index = (y + STEP_SIZE) + column * PADDED_HEIGHT;
   
    uint32_t word;
    cudaMemcpy(&word, &buffer[index], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    word |= bit_mask;
    cudaMemcpy(&buffer[index], &word, sizeof(uint32_t), cudaMemcpyHostToDevice);
}

bool get_cell(uint32_t *buffer, int x, int y) {
    int column = x / 32 + 1;
    uint32_t bit_mask = 0x80000000 >> (x % 32);
    size_t index = (y + STEP_SIZE) + column * PADDED_HEIGHT;
   
    uint32_t word;
    cudaMemcpy(&word, &buffer[index], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return (word & bit_mask) != 0;
}

int main() {
    uint32_t *buffer, *buffer_aux;
    size_t bytes = BUFFER_SIZE * sizeof(uint32_t);
    cudaMalloc(&buffer, bytes);
    cudaMalloc(&buffer_aux, bytes);
    cudaMemset(buffer, 0, bytes);
    cudaMemset(buffer_aux, 0, bytes);

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Setup glider at (1000, 1000)
    printf("\nSetting up glider pattern at (1000, 1000)...\n");
    set_cell(buffer, 1001, 1000);
    set_cell(buffer, 1002, 1001);
    set_cell(buffer, 1000, 1002);
    set_cell(buffer, 1001, 1002);
    set_cell(buffer, 1002, 1002);

    // Run simulation
    printf("Running %d iterations...\n", ITERATIONS);
   
    dim3 grid(HORIZONTAL_GROUPS, VERTICAL_GROUPS, 1);
    dim3 block(1, WORK_GROUP_SIZE, 1);
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int remaining = ITERATIONS;
    while (remaining > 0) {
        int chunk = (remaining > STEP_SIZE) ? STEP_SIZE : remaining;
       
        kernelLOP3<<<grid, block, 0, stream>>>(buffer, buffer_aux, chunk,0);
       
        // Swap buffers
        uint32_t *temp = buffer;
        buffer = buffer_aux;
        buffer_aux = temp;
       
        remaining -= chunk;
    }
   
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
   
    printf("Completed in %.2f ms\n", milliseconds);
    printf("Performance: %.2f iterations/sec\n", ITERATIONS / (milliseconds / 1000.0));
    printf("Throughput: %.2f billion cells/sec\n",
           (double)WIDTH * HEIGHT * ITERATIONS / (milliseconds / 1000.0) / 1e9);

    // Check result
    printf("\nChecking for live cells around glider position...\n");
    bool found = false;
    for (int y = 900; y < 1300 && !found; y++) {
        for (int x = 900; x < 1300; x++) {
            if (get_cell(buffer, x, y)) {
                printf("Found live cell at (%d, %d)\n", x, y);
                found = true;
                break;
            }
        }
    }

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(buffer);
    cudaFree(buffer_aux);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\nDone!\n");
    return 0;
}