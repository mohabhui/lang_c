/**
 * @file rabin_karp_cuda.cu
 * @brief 
 * @version 0.1
 * @date 2022-08-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "float.h"
#include "cstdio"

#define N 1048576
#define N_BLOCKS 1024
#define BLOCK_SIZE 1024

__device__ float counts[N_BLOCKS * BLOCK_SIZE];

/**
 * @brief 
 * 
 */
__global__ void rabin_karp(char *text, char *pattern, int text_len, int pat_len, int chunk_size)
{
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int i;
    int j;

    int base = 10;
    int divisor = 17;

    int ph = 0; //pattern hash
    int th = 0; // text hash
    int bh = 1; // base hash
    int start = threadID * chunk_size;

    counts[threadID] = 0;

    __syncthreads();

    if(start + chunk_size + pat_len > text_len) {
        return;
    }
    
    // Get hash value of bases of any (m-1)-length-text. Used in rolling hash calculation
    for (i = 0; i < pat_len - 1; i++)
        bh = (bh * base) % divisor;

    // Get hash value for pattern and m-length-text at the beginning of the text
    for (i = 0; i < pat_len; i++) {
        ph = (base * ph + pattern[i]) % divisor;
        th = (base * th + text[i + start]) % divisor;
    }

    // Find the match
    for (i = start; i < start + chunk_size; i++) {//loop1 START
        if (ph == th) {
            for (j = 0; j < pat_len; j++) {
                if (text[i + j] != pattern[j])
                break;
            }

            if (j == pat_len){//j will be equal to pat_len if all characters match
                // printf("Match at position:  %d \n", i + 1);
                counts[threadID] = counts[threadID] + 1;
            }
        }

        // Rolling hash calculation
        if (i < text_len - pat_len) {
            th = (base * (th - text[i] * bh) + text[i + pat_len]) % divisor;
            if (th < 0)
                th = (th + divisor);
        }
    }//loop1 END

    return;
}

__global__ void init_kernel(int *count)
{
    for(int i = 0; i < N_BLOCKS * BLOCK_SIZE; i++) {
        counts[i] = 0;
    }
    *count = 0;
    return;
}

// First step in binary reduction:
__global__ void reduction1 (int *summed_counts)
{
    __shared__ int local_counts[BLOCK_SIZE];

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    local_counts[threadIdx.x] = counts[i];

    // To make sure all threads in a block have the count[] value:
    __syncthreads();

    int nTotalThreads = blockDim.x;  // Total number of active threads;
    // only the first half of the threads will be active.

    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);     // divide by two

        if (threadIdx.x < halfPoint)
        {
            int thread2 = threadIdx.x + halfPoint;
            local_counts[threadIdx.x] = local_counts[threadIdx.x] + local_counts[thread2];
        }   
        __syncthreads();
        nTotalThreads = halfPoint;  // Reducing the binary tree size by two
    }

    if (threadIdx.x == 0)
    {
        summed_counts[blockIdx.x] = local_counts[0];
    }

    return;
}



// Second step in binary reduction (one block):
__global__ void reduction2 (int *summed_counts, int *count)
{
    __shared__ int local_counts[N_BLOCKS];

    // Copying from global to shared memory:
    local_counts[threadIdx.x] = summed_counts[threadIdx.x];

    // To make sure all threads in a block have the min[] value:
    __syncthreads();

    int nTotalThreads = blockDim.x;  // Total number of active threads;
    // only the first half of the threads will be active.

    while(nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1);     // divide by two

        if (threadIdx.x < halfPoint)
            {
            int thread2 = threadIdx.x + halfPoint;
            local_counts[threadIdx.x] = local_counts[threadIdx.x] + local_counts[thread2];
        }   
        __syncthreads();
        nTotalThreads = halfPoint;  // Reducing the binary tree size by two
    }

    if (threadIdx.x == 0)
    {
        *count = local_counts[0];
    }

    return;
}

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char *argv[])
{
    int total_count = 0;
    int *host_count;
    char *text_dev;    /* array for computation on device */
    char *pattern_dev;
    //char *text = "The Wilfrid Laurier University or the University of Wilfrid Laurier";
    char *pattern;
    int *summed_counts;
    int *count;
    char *text;
    int num_chunks = N_BLOCKS * BLOCK_SIZE;
    long total_chars_processed = 0;
    
    int chunksize;
    size_t memsize;
    size_t patsize;

    text = (char *)malloc(N * sizeof(char));
    pattern = (char *)malloc(4 * sizeof(char));

    FILE *fp;

    fp = fopen("data/windows_mb512.log", "rb");
    fseek(fp, 0L, SEEK_END);
    size_t file_size = ftell(fp);
    printf("file size: %d\n", file_size);
    rewind(fp);

    // seed random number generator
    //srand(time(NULL));

    //printf("generating random characters\n");

    /*
    for(int i = 0; i < N; i++) {
        text[i] = (char) ('A' + (rand() % 4));
    }

    printf("finished generating characters\n");
    
    printf("text: ");
    printf(text);
    printf("\n");
    printf("pattern: ");
    printf(pattern);
    printf("\n");
    */

    pattern[0] = 'F';
    pattern[1] = 'a';
    pattern[2] = 'i';
    pattern[3] = 'l';
    pattern[4] = 'e';
    pattern[5] = 'd';

    // assuming the length of the string is evenly divided by the block size and
    // number

    host_count = (int *)malloc(sizeof(int));

    /* allocate arrays on device */

    cudaMalloc((void **) &text_dev, memsize);
    cudaMalloc((void **) &pattern_dev, patsize);
    cudaMalloc((void **) &summed_counts, N_BLOCKS * sizeof(int));
    cudaMalloc((void **) &count, sizeof(int));
     
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    for( int i = 0; i < file_size; i += N) {

        fread(text, N, sizeof(char), fp);

        chunksize = strlen(text) / num_chunks;
        memsize = strlen(text) * sizeof(char);
        patsize = strlen(pattern) * sizeof(char);

        printf("text size: %d \n", memsize);
        printf("pattern size: %d \n", patsize);
        printf("chunk size: %d \n", chunksize);

        printf("Copying text to device\n");
        /* copy arrays to device memory (synchronous) */
        cudaMemcpy(text_dev, text, memsize, cudaMemcpyHostToDevice);
        cudaMemcpy(pattern_dev, pattern, patsize, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        printf("Initializing Kernel\n");
        init_kernel<<<1, 1>>> (count);

        printf("Executing Pattern Matching\n");
        rabin_karp <<< N_BLOCKS, BLOCK_SIZE >>> (text_dev, pattern_dev, memsize, patsize, chunksize);

        cudaDeviceSynchronize();

        printf("Executing first level of reduction\n");
        // First level binary reduction:
        reduction1 <<< N_BLOCKS, BLOCK_SIZE >>> (summed_counts);

        cudaDeviceSynchronize();

        printf("Executing second level of reduction\n");
        // Second level binary reduction (only one block):
        reduction2 <<< 1, N_BLOCKS >>> (summed_counts, count);

        cudaDeviceSynchronize();

        cudaMemcpy(host_count, count, sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        total_chars_processed += N;
        total_count += *host_count;
        // print out the results
        printf("total number of pattern matches found in chunk: %d\n", *host_count);
        printf("total characters processed: %d\n", total_chars_processed);
        printf("total patterns found: %d\n", total_count);
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("total execution time for 512MB file: %f", cpu_time_used);

    fclose(fp);

    // free up memory
    cudaFree(text_dev);
    cudaFree(pattern_dev);
    cudaFree(summed_counts);
    cudaFree(count);
    free(host_count);

    return 0;
}