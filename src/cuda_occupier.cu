#include <cstdio>
#include <signal.h>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <cuda_runtime.h>

const size_t ARRAY_SIZE = (512 * 1024 * 1024) / sizeof(float);
const int THREADS_PER_BLOCK = 16;
const int NUM_BLOCKS = 16; // Fixed number of blocks

static void sigdown(int signo) {
    psignal(signo, "Shutting down, got signal");
    exit(0);
}

static void sigreap(int signo) {
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

__global__ void initialize(float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = THREADS_PER_BLOCK * NUM_BLOCKS;
    size_t size_per_thread = ARRAY_SIZE / total_threads;

    for (int i = idx * size_per_thread; i < (idx + 1) * size_per_thread; i += 1) {
        data[i] = i / (1.0 * ARRAY_SIZE);
    }
}

__global__ void sum_array(const float* data, size_t size, float* result) {
    __shared__ float partial_sum[THREADS_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_threads = THREADS_PER_BLOCK * NUM_BLOCKS;
    size_t size_per_thread = ARRAY_SIZE / total_threads;

    float local_sum = 0.0f;
    for (int i = idx * size_per_thread; i < (idx + 1) * size_per_thread; i += 1) {
        local_sum += data[i];
    }

    partial_sum[tid] = local_sum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, partial_sum[0]);
    }
}

int main() {
    struct sigaction sa_int;
    memset(&sa_int, 0, sizeof(sa_int));
    sa_int.sa_handler = sigdown;
    if (sigaction(SIGINT, &sa_int, NULL) < 0) {
        return 1;
    }

    struct sigaction sa_term;
    memset(&sa_term, 0, sizeof(sa_term));
    sa_term.sa_handler = sigdown;
    if (sigaction(SIGTERM, &sa_term, NULL) < 0) {
        return 2;
    }

    struct sigaction sa_chld;
    memset(&sa_chld, 0, sizeof(sa_chld));
    sa_chld.sa_handler = sigreap;
    sa_chld.sa_flags = SA_NOCLDSTOP;
    if (sigaction(SIGCHLD, &sa_chld, NULL) < 0) {
        return 3;
    }

    // Allocate GPU memory
    float* d_data;
    float* d_result;
    cudaMalloc(&d_data, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    while (true) {
        initialize << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (d_data, ARRAY_SIZE);
        cudaDeviceSynchronize();

        float result = 0.0f;
        cudaMemcpy(d_result, &result, sizeof(float), cudaMemcpyHostToDevice);
        sum_array << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (d_data, ARRAY_SIZE, d_result);
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        fprintf(stdout, "Sum of array: %f\n", result);

        sleep(10);
    }

    cudaFree(d_data);
    cudaFree(d_result);

    return 42;
}
