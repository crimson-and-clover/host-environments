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
#include <nvml.h>
#include <cmath>
#include <chrono>

const double TARGET_GPU_UTIL = 0.25;
const int INIT_BLOCK_COUNT = 4000;
const int THREADS_PER_BLOCK = 32;
const size_t ARRAY_SIZE_PER_THREADS = 100;

#define CHECK_CUDA(A) \
cuda_err = A; \
if (cuda_err != cudaSuccess) { \
    fprintf(stderr, "%s: %s at line %d\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err), __LINE__); \
}

static void sigdown(int signo) {
    psignal(signo, "Shutting down, got signal");
    exit(0);
}

static void sigreap(int signo) {
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

__global__ void workload(float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    size_t size_per_thread = (size + total_threads - 1) / total_threads;
    if (size_per_thread <= 0) {
        size_per_thread = 1;
    }
    int64_t lower_bound = idx * size_per_thread;
    int64_t upper_bound = (idx + 1) * size_per_thread > size ? size : (idx + 1) * size_per_thread;
    if (lower_bound >= size) {
        return;
    }

    for (int64_t i = lower_bound; i < upper_bound; i += 1) {
        data[i] = 1.0 * (i - lower_bound) / (upper_bound - lower_bound);
    }
    float sum = 0.0f;
    for (int i = 0; i < 1000; i += 1) {
        for (int64_t j = lower_bound; j < upper_bound; j += 1) {
            sum *= data[j];
        }
    }
}

int64_t get_milliseconds() {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());
    return duration.count();
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

    //避免cuda上下文导致初始化失败
    int current_device = 0;
    cudaError_t cuda_err;
    int device_count = 0;
    for (int device = 1; device < 10; device += 1) {
        auto pid = fork();
        if (pid == -1) {
            fprintf(stderr, "Failed to fork\n");
            exit(1);
        }
        if (pid == 0) {
            current_device = device;
            break;
        }
    }

    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (cuda_err != cudaSuccess) {
        exit(1);
    }
    if (current_device >= device_count) {
        exit(0);
    }

    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(current_device, &device);


    CHECK_CUDA(cudaSetDevice(current_device));
    if (cuda_err != cudaSuccess) {
        exit(1);
    }

    float* cuda_data = nullptr;
    size_t cuda_data_size = 0;
    int total_blocks = INIT_BLOCK_COUNT;
    int64_t start_time;
    while (true) {
        cudaDeviceProp device_prop;
        CHECK_CUDA(cudaGetDeviceProperties(&device_prop, current_device));
        if (cuda_err != cudaSuccess) {
            continue;
        }
        size_t free_mem, total_mem;
        CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
        if (cuda_err != cudaSuccess) {
            continue;
        }
        double mem_usage = (total_mem - free_mem) / (1.0 * total_mem);

        nvmlUtilization_t utilization;
        nvmlDeviceGetUtilizationRates(device, &utilization);
        double gpu_util = 1.0 * utilization.gpu / 100;

        if (cuda_data_size == 0 || get_milliseconds() - start_time > 5000) {
            fprintf(stderr, "Device %d (%s) - Free Memory: %" PRId64 " MB, Total Memory: %" PRId64 " MB, Memory Usage: %d%%, Gpu Usage: %.0f%%\n",
                current_device,
                device_prop.name,
                int64_t(free_mem / 1024 / 1024),
                int64_t(total_mem / 1024 / 1024),
                int(mem_usage * 100),
                gpu_util * 100
            );
        }

        //如果内存占用大于0.5，说明有进程需要用gpu，那么退出
        if (mem_usage > 0.3) {
            goto sleep;
        }

        //如果没有占用内存，利用率已经达标，那么退出
        if (cuda_data_size == 0 && gpu_util > TARGET_GPU_UTIL) {
            goto sleep;
        }

        //开始占用gpu
        if (cuda_data_size == 0) {
            fprintf(stderr, "Device %d under threshold, occupy device\n", current_device);
            int total_threads = total_blocks * THREADS_PER_BLOCK;
            size_t array_size = total_threads * ARRAY_SIZE_PER_THREADS;
            cuda_data_size = array_size;
            cudaMalloc(&cuda_data, cuda_data_size * sizeof(float));
            start_time = get_milliseconds();
        }

        //每10秒，更新thread总数，调整gpu利用率
        if (cuda_data_size > 0 && get_milliseconds() - start_time > 5000) {
            double workload_ratio = sqrt(TARGET_GPU_UTIL / (gpu_util + 0.01));
            total_blocks = int(total_blocks * workload_ratio);
            fprintf(stderr, "Update Device %d total blocks %d\n", current_device, total_blocks);
            if (total_blocks > 1000000) {
                total_blocks = 1000000;
            }
            if (total_blocks <= 0) {
                total_blocks = INIT_BLOCK_COUNT;
                goto sleep;
            }
            int total_threads = total_blocks * THREADS_PER_BLOCK;
            size_t array_size = total_threads * ARRAY_SIZE_PER_THREADS;
            if (cuda_data_size != array_size) {
                if (cuda_data_size != 0) {
                    cudaFree(cuda_data);
                    cuda_data = nullptr;
                }
                cuda_data_size = array_size;
                cudaMalloc(&cuda_data, cuda_data_size * sizeof(float));
            }
            start_time = get_milliseconds();
        }

        {
            workload << <total_blocks, THREADS_PER_BLOCK >> > (cuda_data, cuda_data_size);
        }

        continue;

    sleep:
        if (cuda_data_size != 0) {
            fprintf(stderr, "Device %d reach threshold, free device\n", current_device);
            cudaFree(cuda_data);
            cuda_data = nullptr;
            cuda_data_size = 0;
        }
        sleep(10);
    }
    nvmlShutdown();
    return 42;
}
