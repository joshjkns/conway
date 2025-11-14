#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

const int ITERATIONS = 100000;
const int WIDTH = 16384;
const int HEIGHT = 16384;

void iterationsRun(uint64_t **current_ptr, uint64_t **next_ptr, int width, int height, int iterations);
void gol(uint64_t *current_ptr, int width, int height, int iterations);

// Initialize bit-packed grid with random pattern
void init_grid_bitpacked(uint64_t *grid, int width, int height, float density) {
    int xElements = width / 64;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < xElements; x++) {
            uint64_t word = 0;
            for (int bit = 0; bit < 64; bit++) {
                if ((float)rand() / RAND_MAX < density) {
                    word |= (1ULL << bit);
                }
            }
            grid[y * xElements + x] = word;
        }
    }
}

uint64_t* run_benchmark_cuda() {
    int width = WIDTH;
    int height = HEIGHT;
    int iterations = ITERATIONS;
    int xElements = width / 64;
    size_t grid_size = xElements * height * sizeof(uint64_t);

    uint64_t *grid1 = (uint64_t*)aligned_alloc(64, grid_size);
    uint64_t *grid2 = (uint64_t*)aligned_alloc(64, grid_size);

    if (!grid1 || !grid2) {
        fprintf(stderr, "Memory allocation failed (CUDA benchmark)\n");
        free(grid1);
        free(grid2);
        return NULL;
    }

    srand(42);
    init_grid_bitpacked(grid1, width, height, 0.15f);
    memset(grid2, 0, grid_size);

    uint64_t *current = grid1;
    uint64_t *next = grid2;

    // Warmup
    iterationsRun(&current, &next, width, height, 5);

    // Reset
    current = grid1;
    next = grid2;

    double start = omp_get_wtime();
    iterationsRun(&current, &next, width, height, iterations);
    double end = omp_get_wtime();

    double time_sec = end - start;
    double cells_per_sec = ((double)width * height * iterations) / time_sec;
    double t_cells_per_sec = cells_per_sec / 1000000000000;
    double iterations_per_sec = iterations / time_sec;
    printf("Time taken: %f\n", time_sec);
    printf("Trillion Cells Per Second: %f\n", t_cells_per_sec);
    printf("Iterations Per Second: %f\n", iterations_per_sec);

    return current;
}

uint64_t* run_benchmark_cuda_advanced(){
    int width = WIDTH;
    int height = HEIGHT;
    int iterations = ITERATIONS;
    int xElements = width / 64;
    size_t grid_size = xElements * height * sizeof(uint64_t);
    
    uint64_t *grid1 = (uint64_t*)aligned_alloc(64, grid_size);
    
    if (!grid1) {
        fprintf(stderr, "Memory allocation failed\n");
        free(grid1);
        return NULL;
    }
    
    // Initialize with random pattern
    srand(42);
    init_grid_bitpacked(grid1, width, height, 0.15);
    
    uint64_t *current = grid1;

    // Warmup
    gol(current, xElements, height, 10);
    
    // Reset
    current = grid1;
    
    // Benchmark
    double start = omp_get_wtime();
    gol(current, xElements, height, iterations);
    double end = omp_get_wtime();
    printf("START: %f\n END: %f", start, end);
    
    double time_sec = end - start;
    double cells_per_sec = ((double)width * height * iterations) / time_sec;
    double t_cells_per_sec = cells_per_sec / 1000000000000;
    double iterations_per_sec = iterations / time_sec;
    printf("Time taken: %f\n", time_sec);
    printf("Trillion Cells Per Second: %f\n", t_cells_per_sec);
    printf("Iterations Per Second: %f\n", iterations_per_sec);
    
    return current;
}


int main() {
    uint64_t* base = run_benchmark_cuda();
    uint64_t* advanced = run_benchmark_cuda_advanced();
    int count = 0;
    for (int y = 0; y < 16384; y++){
        for (int x = 0; x < 256; x++){
            if (base[(y * 256) + x] != advanced[(y * 256) + x]){
                count++;
            }
        }
    }
    printf("\n%zu\n", sizeof(*base));
    printf("%zu\n", sizeof(*advanced));
    printf("should be zero: %d", count);
    free(base);
    free(advanced);
    return 0;
}

// nvcc -Xcompiler -fopenmp benchmarkNew.c warpHalo.cu -o golBenchmark
// nvcc -O3 -arch=sm_80 -Xcompiler -fopenmp benchmarkNew.c warpHalo.cu gol.cu -o golBenchmark -use_fast_math
