#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Naive byte-based GOL (for comparison)
void gol_naive(uint8_t **current_ptr, uint8_t **next_ptr, int width, int height, int iterations) {
    uint8_t *current = *current_ptr;
    uint8_t *next = *next_ptr;
    
    for (int iteration = 0; iteration < iterations; iteration++) {
        //#pragma omp parallel for schedule(runtime)
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
        #pragma omp parallel for schedule(runtime)
        for(int y = 0; y < height; y++){
            int yUp = (y - 1 + height) % height;
            int yDown = (y + 1) % height;
            for (int x = 0; x < xElements; x++){
                int x_left = (x - 1 + xElements) % xElements;
                int x_right = (x + 1) % xElements;

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
                next[y * xElements + x] = three | (two & center);
            }
        }
        uint64_t *tmp = current;
        current = next;
        next = tmp;
    }
    *current_ptr = current;
    *next_ptr = next;
}

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

// Initialize byte grid with random pattern
void init_grid_naive(uint8_t *grid, int width, int height, float density) {
    for (int i = 0; i < width * height; i++) {
        grid[i] = ((float)rand() / RAND_MAX < density) ? 1 : 0;
    }
}

// Run benchmark for bit-parallel version
typedef struct {
    const char *name;
    int width;
    int height;
    int iterations;
    double time_sec;
    double cells_per_sec;
    double iterations_per_sec;
    double speedup;
} BenchResult;

BenchResult run_benchmark_bitpacked(int width, int height, int iterations) {
    BenchResult result = {"Bit-parallel", width, height, iterations, 0, 0, 0, 0};
    
    int xElements = width / 64;
    size_t grid_size = xElements * height * sizeof(uint64_t);
    
    uint64_t *grid1 = (uint64_t*)aligned_alloc(64, grid_size);
    uint64_t *grid2 = (uint64_t*)aligned_alloc(64, grid_size);
    
    if (!grid1 || !grid2) {
        fprintf(stderr, "Memory allocation failed\n");
        free(grid1);
        free(grid2);
        return result;
    }
    
    // Initialize with random pattern
    srand(42);
    init_grid_bitpacked(grid1, width, height, 0.15);
    memset(grid2, 0, grid_size);
    
    uint64_t *current = grid1;
    uint64_t *next = grid2;
    
    // Warmup
    gol(&current, &next, width, height, 10);
    
    // Reset
    current = grid1;
    next = grid2;
    
    // Benchmark
    double start = omp_get_wtime();
    gol(&current, &next, width, height, iterations);
    double end = omp_get_wtime();
    
    result.time_sec = end - start;
    result.cells_per_sec = ((double)width * height * iterations) / result.time_sec;
    result.iterations_per_sec = iterations / result.time_sec;
    
    free(grid1);
    free(grid2);
    
    return result;
}

// Run benchmark for naive version
BenchResult run_benchmark_naive(int width, int height, int iterations) {
    BenchResult result = {"Naive", width, height, iterations, 0, 0, 0, 0};
    
    size_t grid_size = width * height * sizeof(uint8_t);
    
    uint8_t *grid1 = (uint8_t*)aligned_alloc(64, grid_size);
    uint8_t *grid2 = (uint8_t*)aligned_alloc(64, grid_size);
    
    if (!grid1 || !grid2) {
        fprintf(stderr, "Memory allocation failed\n");
        free(grid1);
        free(grid2);
        return result;
    }
    
    // Initialize with random pattern
    srand(42);
    init_grid_naive(grid1, width, height, 0.15);
    memset(grid2, 0, grid_size);
    
    uint8_t *current = grid1;
    uint8_t *next = grid2;
    
    // Warmup
    gol_naive(&current, &next, width, height, 10);
    
    // Reset
    current = grid1;
    next = grid2;
    
    // Benchmark
    double start = omp_get_wtime();
    gol_naive(&current, &next, width, height, iterations);
    double end = omp_get_wtime();
    
    result.time_sec = end - start;
    result.cells_per_sec = ((double)width * height * iterations) / result.time_sec;
    result.iterations_per_sec = iterations / result.time_sec;
    
    free(grid1);
    free(grid2);
    
    return result;
}

// Print results table
void print_results(BenchResult *results, int count) {
    printf("\n");
    printf("┌──────────────┬─────────────┬────────┬───────────┬──────────────┬────────────────┬──────────┐\n");
    printf("│ Algorithm    │ Grid Size   │ Iters  │ Time (s)  │ Mcells/sec   │ Iters/sec      │ Speedup  │\n");
    printf("├──────────────┼─────────────┼────────┼───────────┼──────────────┼────────────────┼──────────┤\n");
    
    for (int i = 0; i < count; i++) {
        BenchResult *r = &results[i];
        printf("│ %-12s │ %4dx%-4d   │ %6d │ %9.3f │ %12.1f │ %14.1f │ %7.1fx │\n",
               r->name, r->width, r->height, r->iterations, 
               r->time_sec, r->cells_per_sec / 1e6, r->iterations_per_sec, r->speedup);
    }
    
    printf("└──────────────┴─────────────┴────────┴───────────┴──────────────┴────────────────┴──────────┘\n");
}

int main(int argc, char **argv) {
    printf("Game of Life Benchmark Suite\n");
    printf("=============================\n");
    printf("Threads available: %d\n", omp_get_max_threads());
    printf("Current schedule: %s\n\n", getenv("OMP_SCHEDULE") ?: "default (runtime)");
    
    // Test configurations
    int configs[][3] = {
        {512, 512, 1000},
        {1024, 1024, 1000},
        {2048, 2048, 500},
        {5120, 5120, 100},
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    BenchResult results[num_configs * 2];  // 2x for naive + bitpacked
    
    for (int i = 0; i < num_configs; i++) {
        int width = configs[i][0];
        int height = configs[i][1];
        int iters = configs[i][2];
        
        printf("Testing %dx%d grid, %d iterations...\n", width, height, iters);
        
        // Run naive version
        printf("  Naive: ");
        fflush(stdout);
        results[i * 2] = run_benchmark_naive(width, height, iters);
        printf("%.3f sec (%.1f Mcells/sec)\n", 
               results[i * 2].time_sec, results[i * 2].cells_per_sec / 1e6);
        
        // Run bit-parallel version
        printf("  Bit-parallel: ");
        fflush(stdout);
        results[i * 2 + 1] = run_benchmark_bitpacked(width, height, iters);
        printf("%.3f sec (%.1f Mcells/sec)\n", 
               results[i * 2 + 1].time_sec, results[i * 2 + 1].cells_per_sec / 1e6);
        
        // Calculate speedup
        results[i * 2].speedup = 1.0;  // Baseline
        results[i * 2 + 1].speedup = results[i * 2].time_sec / results[i * 2 + 1].time_sec;
        
        printf("  → Speedup: %.1fx\n\n", results[i * 2 + 1].speedup);
    }
    
    print_results(results, num_configs * 2);
    
    // Calculate average speedup
    double avg_speedup = 0;
    for (int i = 0; i < num_configs; i++) {
        avg_speedup += results[i * 2 + 1].speedup;
    }
    avg_speedup /= num_configs;
    
    printf("\n=== Summary ===\n");
    printf("Average speedup (bit-parallel vs naive): %.1fx\n", avg_speedup);
    printf("Threads used: %d\n\n", omp_get_max_threads());
    
    printf("=== Tuning Tips ===\n");
    printf("To test different schedules:\n");
    printf("  OMP_SCHEDULE='static,8'  ./benchmark\n");
    printf("  OMP_SCHEDULE='static,16' ./benchmark\n");
    printf("  OMP_SCHEDULE='dynamic,8' ./benchmark\n");
    printf("  OMP_SCHEDULE='guided'    ./benchmark\n\n");
    
    printf("To test different thread counts:\n");
    printf("  OMP_NUM_THREADS=4  ./benchmark\n");
    printf("  OMP_NUM_THREADS=8  ./benchmark\n");
    printf("  OMP_NUM_THREADS=16 ./benchmark\n\n");
    
    return 0;
}