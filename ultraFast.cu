#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

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

// __device__ __forceinline__ ResultOneStepReduce oneStepReduce(inputOneStepReduce inp){
//     ResultOneStepReduce output;
//     uint32_t left_top_xor, left_mid_xor, left_top_maj, left_mid_maj;
//     uint32_t right_top_xor, right_mid_xor, right_top_maj, right_mid_maj;
//     uint32_t resultL, resultR;

//     // Process rows
//     #pragma unroll
//     for (int row = 0; row < 16; row++) {
//         int row = row + 1;

//         // Left half
//         const uint32_t a0 = inp.store[0][row - 1] >> 1;
//         const uint32_t a1 = inp.store[0][row - 1];
//         const uint32_t a2 = __funnelshift_l(inp.store[1][row - 1], inp.store[0][row - 1], 1);
//         const uint32_t a3 = inp.store[0][row] >> 1;
//         const uint32_t a4 = __funnelshift_l(inp.store[1][row], inp.store[0][row], 1);
//         const uint32_t a5 = inp.store[0][row + 1] >> 1;
//         const uint32_t a6 = inp.store[0][row + 1];
//         const uint32_t a7 = __funnelshift_l(inp.store[1][row + 1], inp.store[0][row + 1], 1);

//         if (row == 0) {
//             asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(left_top_xor) : "r"(a2), "r"(a1), "r"(a0));
//             asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(left_mid_xor) : "r"(a4), "r"(a3), "r"(inp.store[0][row]));
//             asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(left_top_maj) : "r"(a2), "r"(a1), "r"(a0));
//             asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(left_mid_maj) : "r"(a4), "r"(a3), "r"(inp.store[][row]));
//         }

//         uint32_t left_bottom_xor, left_bottom_maj;
//         asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(left_bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
//         asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(left_bottom_maj) : "r"(a7), "r"(a6), "r"(a5));

//         resultL = sub_step(a0, a1, a2, a3, a4, a5, a6, a7,
//                                     left_top_xor, left_bottom_xor, left_top_maj, left_bottom_maj, inp.store[0][row]);
//         left_top_xor = left_mid_xor;
//         left_mid_xor = left_bottom_xor;
//         left_top_maj = left_mid_maj;
//         left_mid_maj = left_bottom_maj;

//         // Right half
//         const uint32_t b0 = __funnelshift_r(inp.store[1][row - 1], inp.store[0][row - 1], 1);
//         const uint32_t b1 = inp.store[1][row - 1];
//         const uint32_t b2 = inp.store[1][row - 1] << 1;
//         const uint32_t b3 = __funnelshift_r(inp.store[1][row], inp.store[0][row], 1);
//         const uint32_t b4 = inp.store[1][row] << 1;
//         const uint32_t b5 = __funnelshift_r(inp.store[1][row + 1], inp.store[0][row + 1], 1);
//         const uint32_t b6 = inp.store[1][row + 1];
//         const uint32_t b7 = inp.store[1][row + 1] << 1;

//         if (row == 0) {
//             asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(right_top_xor) : "r"(b2), "r"(b1), "r"(b0));
//             asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(right_mid_xor) : "r"(b4), "r"(b3), "r"(inp.store[1][row]));
//             asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(right_top_maj) : "r"(b2), "r"(b1), "r"(b0));
//             asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(right_mid_maj) : "r"(b4), "r"(b3), "r"(inp.store[1][row]));
//         }

//         uint32_t right_bottom_xor, right_bottom_maj;
//         asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(right_bottom_xor) : "r"(b7), "r"(b6), "r"(b5));
//         asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(right_bottom_maj) : "r"(b7), "r"(b6), "r"(b5));

//         resultR = sub_step(b0, b1, b2, b3, b4, b5, b6, b7,
//                                     right_top_xor, right_bottom_xor, right_top_maj, right_bottom_maj, inp.store[1][row]);
//         right_top_xor = right_mid_xor;
//         right_mid_xor = right_bottom_xor;
//         right_top_maj = right_mid_maj;
//         right_mid_maj = right_bottom_maj;
//         asm("prmt.b32 %0, %1, %2, 0x1076;" : "=r"(output.store[row]) : "r"(resultL), "r"(resultR));
//     }
//     return output;
// }



__global__ void step_kernel(const uint32_t *field, uint32_t *new_field, uint32_t steps, int warpComputeSize) {
    const size_t py = threadIdx.y * 16;
    const size_t i = (blockIdx.x + 1) * PADDED_HEIGHT + blockIdx.y * SIMULATED_ROWS + py;

    uint32_t left[18];
    uint32_t right[18];

    // Load data
    #pragma unroll
    for (int row = 0; row < 4; row++) {
        uint32_t lx, ly, lz, lw, mx, my, mz, mw, rx, ry, rz, rw;
        load_uint4(&lx, &ly, &lz, &lw, &field[i + row * 4 - PADDED_HEIGHT]);
        load_uint4(&mx, &my, &mz, &mw, &field[i + row * 4]);
        load_uint4(&rx, &ry, &rz, &rw, &field[i + row * 4 + PADDED_HEIGHT]);

        permute(&left[row * 4 + 1], lx, mx);
        permute(&right[row * 4 + 1], mx, rx);
        permute(&left[row * 4 + 2], ly, my);
        permute(&right[row * 4 + 2], my, ry);
        permute(&left[row * 4 + 3], lz, mz);
        permute(&right[row * 4 + 3], mz, rz);
        permute(&left[row * 4 + 4], lw, mw);
        permute(&right[row * 4 + 4], mw, rw);
    }

    // Maybe put it here if really necsary?
    // //-------------------- I think I can remove this---------------------------------
    // if (blockIdx.x == 0) {
    //     #pragma unroll
    //     for (int row = 0; row < WORK_PER_THREAD; row++)
    //         left[row + 1] &= 0x0000FFFF;
    // }
    // if (blockIdx.x == gridDim.x - 1) {
    //     #pragma unroll
    //     for (int row = 0; row < WORK_PER_THREAD; row++)
    //         right[row + 1] &= 0xFFFF0000;
    // }
    // //-------------------------------------------------------------------------------

    // Simulation loop
    for (uint32_t step = 0; step < warpComputeSize; step++) {
        uint32_t result_left[16];
        uint32_t result_right[16];

        // Boundaries
        if (blockIdx.y == 0 && threadIdx.y == CLIP_TOP_LY) {
            left[CLIP_TOP_OFFSET] = 0;
            right[CLIP_TOP_OFFSET] = 0;
        }
        if (blockIdx.y == gridDim.y - 1 && threadIdx.y == CLIP_BOTTOM_LY) {
            left[CLIP_BOTTOM_OFFSET] = 0;
            right[CLIP_BOTTOM_OFFSET] = 0;
        }

        //-------------------- I think I can remove this --------------------------------
        if (blockIdx.x == 0) {
            #pragma unroll
            for (int row = 0; row < 16; row++)
                left[row + 1] &= 0x0000FFFF;
        }
        if (blockIdx.x == gridDim.x - 1) {
            #pragma unroll
            for (int row = 0; row < 16; row++)
                right[row + 1] &= 0xFFFF0000;
        }
        //-------------------------------------------------------------------------------

        // Warp shuffles
        left[0] = __shfl_up_sync(0xFFFFFFFF, left[16], 1);
        right[0] = __shfl_up_sync(0xFFFFFFFF, right[16], 1);
        left[17] = __shfl_down_sync(0xFFFFFFFF, left[1], 1);
        right[17] = __shfl_down_sync(0xFFFFFFFF, right[1], 1);

        uint32_t xor00, xor01, sum00, sum01;
        uint32_t xor10, xor11, sum10, sum11;
        // Setting up xor and sum chains, for the column 0
        {
            const uint32_t BeginA0 = left[0] >> 1;
            const uint32_t BeginA1 = left[0];
            const uint32_t BeginA2 = __funnelshift_l(right[0], left[0], 1);
            const uint32_t BeginA3 = left[1] >> 1;
            const uint32_t BeginA4 = __funnelshift_l(right[1], left[1], 1);

            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(xor00) : "r"(BeginA2), "r"(BeginA1), "r"(BeginA0));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(sum00) : "r"(BeginA2), "r"(BeginA1), "r"(BeginA0));
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(xor01) : "r"(BeginA4), "r"(BeginA3), "r"(left[1]));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(sum01) : "r"(BeginA4), "r"(BeginA3), "r"(left[1]));
        }

        // Setting up xor and sum chains, for the column 1
        {
            const uint32_t BeginB0 = __funnelshift_r(right[0], left[0], 1);
            const uint32_t BeginB1 = right[0];
            const uint32_t BeginB2 = right[0] << 1;
            const uint32_t BeginB3 = __funnelshift_r(right[1], left[1], 1);
            const uint32_t BeginB4 = right[1] << 1;

            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(xor10) : "r"(BeginB2), "r"(BeginB1), "r"(BeginB0));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(sum10) : "r"(BeginB2), "r"(BeginB1), "r"(BeginB0));
            asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(xor11) : "r"(BeginB4), "r"(BeginB3), "r"(right[1]));
            asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(sum11) : "r"(BeginB4), "r"(BeginB3), "r"(right[1]));
        }

        // Doing the required amount of rows of work
        #pragma unroll
        for (int row = 1; row <= 16; row++) {
            // Left half
            {
                const uint32_t a3 = left[row] >> 1;
                const uint32_t a4 = __funnelshift_l(right[row], left[row], 1);
                const uint32_t a5 = left[row + 1] >> 1;
                const uint32_t a6 = left[row + 1];
                const uint32_t a7 = __funnelshift_l(right[row + 1], left[row + 1], 1);


                uint32_t left_bottom_xor, left_bottom_maj;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(left_bottom_xor) : "r"(a7), "r"(a6), "r"(a5));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(left_bottom_maj) : "r"(a7), "r"(a6), "r"(a5));

                uint32_t aA, y2, magic0, magic1, magic2;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(aA) : "r"(left_top_xor), "r"(a4), "r"(a3));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(y2) : "r"(left_top_xor), "r"(a4), "r"(a3));
                asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(left_bottom_xor), "r"(aA), "r"(left[row]));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(left[row]), "r"(y2));
                asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(left_bottom_maj), "r"(left_top_maj));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result_left[row-1]) : "r"(magic2), "r"(magic0), "r"(magic1));

                left_top_xor = left_mid_xor;
                left_mid_xor = left_bottom_xor;
                left_top_maj = left_mid_maj;
                left_mid_maj = left_bottom_maj;
                result_left[row-1] = result;
            }

            // Right half
            {
                const uint32_t b3 = __funnelshift_r(right[row], left[row], 1);
                const uint32_t b4 = right[row] << 1;
                const uint32_t b5 = __funnelshift_r(right[row + 1], left[row + 1], 1);
                const uint32_t b6 = right[row + 1];
                const uint32_t b7 = right[row + 1] << 1;


                uint32_t right_bottom_xor, right_bottom_maj;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(right_bottom_xor) : "r"(b7), "r"(b6), "r"(b5));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(right_bottom_maj) : "r"(b7), "r"(b6), "r"(b5));

                uint32_t bB, x2;
                asm("lop3.b32 %0, %1, %2, %3, 0b10010110;" : "=r"(bB) : "r"(right_top_xor), "r"(b4), "r"(b3));
                asm("lop3.b32 %0, %1, %2, %3, 0b11101000;" : "=r"(x2) : "r"(right_top_xor), "r"(b4), "r"(b3));
                asm("lop3.b32 %0, %1, %2, %3, 0b00111110;" : "=r"(magic0) : "r"(right_bottom_xor), "r"(bB), "r"(right[row]));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011011;" : "=r"(magic1) : "r"(magic0), "r"(right[row]), "r"(x2));
                asm("lop3.b32 %0, %1, %2, %3, 0b10010001;" : "=r"(magic2) : "r"(magic1), "r"(right_bottom_maj), "r"(right_top_maj));
                asm("lop3.b32 %0, %1, %2, %3, 0b01011000;" : "=r"(result_right[row-1]) : "r"(magic2), "r"(magic0), "r"(magic1));

                right_top_xor = right_mid_xor;
                right_mid_xor = right_bottom_xor;
                right_top_maj = right_mid_maj;
                right_mid_maj = right_bottom_maj;
            }
        }

        // Copy results after as loop still going, os can't overwrite values yet
        #pragma unroll
        for (int row = 0; row < 16; row++) {
            left[row+1] = result_left[row];
            right[row+1] = result_right[row];
        }
    }

    // Write back
    #pragma unroll
    for (int row = 0; row < 16; row++) {
        if (py + row >= STEP_SIZE && py + row < WORK_GROUP_SIZE * 16 - STEP_SIZE) {
            permute(&new_field[i + row], left[row + 1], right[row + 1]);
        }
    }
}

// ============================================================================
// Main Program
// ============================================================================

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
    printf("Game of Life: %dx%d grid, %d iterations\n", WIDTH, HEIGHT, ITERATIONS);
    printf("Grid dimensions: (%d, %d, 1)\n", HORIZONTAL_GROUPS, VERTICAL_GROUPS);
    printf("Block dimensions: (1, %d, 1)\n", WORK_GROUP_SIZE);
    printf("Buffer size: %d uint32s = %.2f MB per buffer\n",
           BUFFER_SIZE, BUFFER_SIZE * 4.0 / (1024*1024));

    // Allocate GPU memory
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
       
        step_kernel<<<grid, block, 0, stream>>>(buffer, buffer_aux, chunk);
       
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




// ============================================================================
// Simple CPU Reference - Game of Life Rules
// ============================================================================

int count_neighbors_cpu(uint8_t *grid, int width, int height, int x, int y) {
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                if (grid[ny * width + nx]) count++;
            }
        }
    }
    return count;
}

void step_cpu(uint8_t *grid, int width, int height) {
    uint8_t *next = (uint8_t*)calloc(width * height, sizeof(uint8_t));
   
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int n = count_neighbors_cpu(grid, width, height, x, y);
            bool alive = grid[y * width + x];
           
            if ((alive && (n == 2 || n == 3)) || (!alive && n == 3)) {
                next[y * width + x] = 1;
            }
        }
    }
   
    memcpy(grid, next, width * height);
    free(next);
}

// ============================================================================
// Test Functions
// ============================================================================

void test_correctness() {
    printf("\n=== CORRECTNESS TEST ===\n");
   
    int size = 128;
   
    // CPU version
    uint8_t *cpu_grid = (uint8_t*)calloc(size * size, sizeof(uint8_t));
   
    // Set up glider on CPU
    cpu_grid[10 * size + 11] = 1;  // (11, 10)
    cpu_grid[11 * size + 12] = 1;  // (12, 11)
    cpu_grid[12 * size + 10] = 1;  // (10, 12)
    cpu_grid[12 * size + 11] = 1;  // (11, 12)
    cpu_grid[12 * size + 12] = 1;  // (12, 12)
   
    // GPU version
    uint32_t *buffer, *buffer_aux;
    size_t bytes = BUFFER_SIZE * sizeof(uint32_t);
    cudaMalloc(&buffer, bytes);
    cudaMalloc(&buffer_aux, bytes);
    cudaMemset(buffer, 0, bytes);
    cudaMemset(buffer_aux, 0, bytes);
   
    cudaStream_t stream;
    cudaStreamCreate(&stream);
   
    // Set up same glider on GPU
    set_cell(buffer, 11, 10);
    set_cell(buffer, 12, 11);
    set_cell(buffer, 10, 12);
    set_cell(buffer, 11, 12);
    set_cell(buffer, 12, 12);
   
    // Run both for 20 steps
    printf("Running 20 steps...\n");
   
    for (int i = 0; i < 20; i++) {
        step_cpu(cpu_grid, size, size);
    }
   
    dim3 grid(HORIZONTAL_GROUPS, VERTICAL_GROUPS, 1);
    dim3 block(1, WORK_GROUP_SIZE, 1);
   
    int remaining = 20;
    uint32_t *buf = buffer;
    uint32_t *aux = buffer_aux;
    while (remaining > 0) {
        int chunk = (remaining > STEP_SIZE) ? STEP_SIZE : remaining;
        step_kernel<<<grid, block, 0, stream>>>(buf, aux, chunk);
        uint32_t *temp = buf;
        buf = aux;
        aux = temp;
        remaining -= chunk;
    }
    cudaStreamSynchronize(stream);
   
    // Copy back if needed
    if (buf != buffer) {
        cudaMemcpy(buffer, buf, bytes, cudaMemcpyDeviceToDevice);
    }
   
    // Compare
    printf("Comparing results...\n");
    int mismatches = 0;
    int cpu_alive = 0;
    int gpu_alive = 0;
   
    for (int y = 0; y < size && mismatches < 20; y++) {
        for (int x = 0; x < size && mismatches < 20; x++) {
            bool cpu_val = cpu_grid[y * size + x];
            bool gpu_val = get_cell(buffer, x, y);
           
            if (cpu_val) cpu_alive++;
            if (gpu_val) gpu_alive++;
           
            if (cpu_val != gpu_val) {
                printf("  ❌ Mismatch at (%d, %d): CPU=%d, GPU=%d\n",
                       x, y, cpu_val, gpu_val);
                mismatches++;
            }
        }
    }
   
    printf("\nCPU alive cells: %d\n", cpu_alive);
    printf("GPU alive cells: %d\n", gpu_alive);
   
    if (mismatches == 0) {
        printf("\n✅ SUCCESS! All cells match!\n");
    } else {
        printf("\n❌ FAILED! Found %d+ mismatches\n", mismatches);
    }
   
    free(cpu_grid);
    cudaFree(buffer);
    cudaFree(buffer_aux);
    cudaStreamDestroy(stream);
}

// ============================================================================
// Simple Main - Just Calls the Test
// ============================================================================

// int main() {
//     printf("========================================\n");
//     printf("Testing CUDA Game of Life Implementation\n");
//     printf("========================================\n");
   
//     test_correctness();
   
//     return 0;
// }

/*
=============================================================================
HOW TO USE:
=============================================================================

1. Copy everything above
2. Paste at the END of your game_of_life.cu file
3. Compile:
   
   nvcc -O3 game_of_life.cu -o test
   
4. Run:
   
   ./test

=============================================================================
EXPECTED OUTPUT IF CORRECT:
=============================================================================

========================================
Testing CUDA Game of Life Implementation
========================================

=== CORRECTNESS TEST ===
Running 20 steps...
Comparing results...

CPU alive cells: 5
GPU alive cells: 5

✅ SUCCESS! All cells match!

=============================================================================
IF WRONG, YOU'LL SEE:
=============================================================================

❌ Mismatch at (25, 24): CPU=1, GPU=0
❌ Mismatch at (26, 25): CPU=1, GPU=0
...

CPU alive cells: 5
GPU alive cells: 3

❌ FAILED! Found 10+ mismatches

=============================================================================
*/