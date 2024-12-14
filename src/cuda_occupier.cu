#include <cstdio>
#include <signal.h>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <inttypes.h>
#include <vector>
#include <cuda_runtime.h>

const int THREADS_PER_BLOCK = 64;
const int NUM_BLOCKS = 64;

#define CHECK_CUDA(A) \
cuda_err = A; \
if (cuda_err != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err)); \
    fprintf(stderr, "\tat line %d", __LINE__); \
}

static void sigdown(int signo) {
    psignal(signo, "Shutting down, got signal");
    exit(0);
}

static void sigreap(int signo) {
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

struct CudaWorkspace {
    float* data;
    size_t size;
    float* result;
};

__global__ void initialize(float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    size_t size_per_thread = (size + total_threads - 1) / total_threads;
    if (size_per_thread == 0) {
        size_per_thread = 1;
    }

    for (int64_t i = idx * size_per_thread; i < (idx + 1) * size_per_thread && i < size; i += 1) {
        if (size % 2 == 1 && i == size - 1) {
            data[i] = 0;
        }
        else if (i % 2 == 0) {
            data[i] = -1;
        }
        else if (i % 2 == 1) {
            data[i] = 1;
        }
    }
}

__global__ void sum_array(const float* data, size_t size, float* result) {
    __shared__ float partial_sum[THREADS_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    size_t size_per_thread = (size + total_threads - 1) / total_threads;
    if (size_per_thread == 0) {
        size_per_thread = 1;
    }

    float local_sum = 0.0f;
    for (int64_t i = idx * size_per_thread; i < (idx + 1) * size_per_thread && i < size; i += 1) {
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

    cudaError_t cuda_err;
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (cuda_err != cudaSuccess) {
        exit(1);
    }

    std::vector<CudaWorkspace> workspaces;
    for (int device = 0; device < device_count; device += 1) {
        workspaces.push_back(CudaWorkspace{
            .data = nullptr,
            .size = 0,
            .result = nullptr
            });
    }

    while (true) {
        for (int device = 0; device < device_count; ++device) {
            CHECK_CUDA(cudaSetDevice(device));
            if (cuda_err != cudaSuccess) {
                continue;
            }

            auto workspace = workspaces[device];

            cudaDeviceProp device_prop;
            CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device));
            if (cuda_err != cudaSuccess) {
                continue;
            }

            size_t free_mem, total_mem;
            CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
            if (cuda_err != cudaSuccess) {
                continue;
            }

            double mem_usage = (total_mem - free_mem) / (1.0 * total_mem);
            fprintf(stderr, "Device %d (%s) - Free Memory: %" PRId64 " MB, Total Memory: %" PRId64 " MB, Usage: %d%%\n",
                device,
                device_prop.name,
                int64_t(free_mem / 1024 / 1024),
                int64_t(total_mem / 1024 / 1024),
                int(mem_usage * 100)
            );

            if (mem_usage > 0.5) {
                if (workspace.size > 0) {
                    fprintf(stderr, "Device %d reach threshold, free device\n", device);
                    cudaFree(workspace.data);
                    workspace.data = nullptr;
                    cudaFree(workspace.result);
                    workspace.result = nullptr;
                    workspace.size = 0;
                }
                workspaces[device] = workspace;
                continue;
            }
            if (mem_usage < 0.25) {
                if (workspace.size > 0) {
                    cudaFree(workspace.data);
                    workspace.data = nullptr;
                    cudaFree(workspace.result);
                    workspace.result = nullptr;
                    workspace.size = 0;
                }
                fprintf(stderr, "Device %d under threshold, occupy device\n", device);
                workspace.size = size_t(0.25 * total_mem / sizeof(float));
                cudaMalloc(&workspace.data, workspace.size * sizeof(float));
                cudaMalloc(&workspace.result, sizeof(float));
                workspaces[device] = workspace;
            }

            if (workspace.size == 0) {
                continue;
            }

            initialize << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (workspace.data, workspace.size);
            CHECK_CUDA(cudaDeviceSynchronize());

            float result = 0.0f;
            CHECK_CUDA(cudaMemcpy(workspace.result, &result, sizeof(float), cudaMemcpyHostToDevice));
            sum_array << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (workspace.data, workspace.size, workspace.result);
            CHECK_CUDA(cudaMemcpy(&result, workspace.result, sizeof(float), cudaMemcpyDeviceToHost));

            fprintf(stdout, "Sum of array: %f\n", result);
        }

        sleep(10);
    }

    return 42;
}
