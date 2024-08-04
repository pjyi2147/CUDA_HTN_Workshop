#include <stdio.h>

#define N 16

void random_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 10;
    }
}

__global__ void add(int *a, int *b, int *c)
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main()
{
    int size = sizeof(int) * N;
    int *a = (int *)malloc(size);
    random_ints(a, N);
    int *b = (int *)malloc(size);
    random_ints(b, N);
    int *c = (int *)malloc(size);
    memset(c, 0, size);


    // device copies of a, b, c
    int *d_a, *d_b, *d_c;

    // Allocate memory to device
    // Allocate memory to d_b and d_c using the following example
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);


    //// Copy inputs to device
    //// Copy input to d_b using the following example
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    //// Launch kernel
    add<<<N, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    //// Copy result from device to host
    //// What is the difference of this line compared to above cudaMemcpy?
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    //// print result, is result correct?
    for (int i = 0; i < N; i++)
    {
        printf("i = %d, %d + %d = %d\n", i, a[i], b[i], c[i]);
    }

    //// Memory cleanup!
    //// Clean up memory for d_a, d_b, d_c with following example.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //// Of course the same for host memory, too.
    free(a);
    free(b);
    free(c);

    return 0;
}
