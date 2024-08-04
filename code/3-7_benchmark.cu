
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

#define N 400000000

#define BLOCK_SIZE 1024

void random_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 10;
    }
}

__global__ void sub(int *a, int *c, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // check the difference here
    if (idx < len)
    {
        c[idx] = a[idx] - a[idx];
    }
}

void sub_cpu(int *a, int *c, int len)
{
    for (int i = 0; i < len; i++)
    {
        c[i] = a[i] - a[i];
    }
}

void gpu(int *a, int *c, int len)
{
    size_t size = sizeof(int) * N;
    // device copies of a, c
    int *d_a, *d_c;

    // Allocate memory to device
    // Allocate memory to d_b and d_c using the following example
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_c, size);

    //// Copy inputs to device
    //// Copy input to d_b using the following example
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    //// Launch kernel
    //// what's the difference here?
    int NUMBLOCKS = (N - 1) / BLOCK_SIZE + 1;

    struct timeval gpustart, gpuend;
    gettimeofday(&gpustart, NULL);
    sub<<<NUMBLOCKS, BLOCK_SIZE>>>(d_a, d_c, N);
    cudaDeviceSynchronize();
    gettimeofday(&gpuend, NULL);

    double secs = (double)(gpuend.tv_usec - gpustart.tv_usec) / 1000000 + (double)(gpuend.tv_sec - gpustart.tv_sec);
    printf("gpu computation time taken %f s\n", secs);

    //// Copy result from device to host
    //// What is the difference of this line compared to above cudaMemcpy?
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    //// Clean up memory for d_a, d_b, d_c with following example
    cudaFree(d_a);
    cudaFree(d_c);
}

int main()
{
    size_t size = sizeof(int) * N;
    int *a = (int *)malloc(size);
    random_ints(a, N);
    int *c = (int *)malloc(size);
    random_ints(c, N);

    // benchmark CPU (one thread)
    struct timeval cpustart, cpuend;
    gettimeofday(&cpustart, NULL);
    sub_cpu(a, c, N);
    gettimeofday(&cpuend, NULL);

    for (int i = 0; i < N; i++)
    {
        assert(c[i] == 0);
    }

    double secs = (double)(cpuend.tv_usec - cpustart.tv_usec) / 1000000 + (double)(cpuend.tv_sec - cpustart.tv_sec);
    printf("cpu computation time taken %f s\n", secs);

    // reset c with random values
    random_ints(c, N);

    // benchmark GPU
    struct timeval gpustart, gpuend;
    gettimeofday(&gpustart, NULL);
    gpu(a, c, N);
    gettimeofday(&gpuend, NULL);

    for (int i = 0; i < N; i++)
    {
        assert(c[i] == 0);
    }

    secs = (double)(gpuend.tv_usec - gpustart.tv_usec) / 1000000 + (double)(gpuend.tv_sec - gpustart.tv_sec);
    printf("gpu time taken %f s\n", secs);

    //// Memory cleanup!
    free(a);
    free(c);

    return 0;
}
