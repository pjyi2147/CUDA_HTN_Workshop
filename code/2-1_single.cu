#include <stdio.h>
__global__ void add(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main()
{
    int a = 3, b = 2, c = 0;    // host copies of a, b, c
    int *d_a, *d_b, *d_c;       // device copies of a, b, c
    int size = sizeof(int);

    //// Allocate memory to device
    //// Allocate memory to d_a, d_b, and d_c using the following example
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    //// Copy a, b to device
    //// Copy a, b to d_a, d_b using the following example
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    //// Launch kernel
    add<<<1, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    //// Copy result from device (d_c) to host (c)
    //// What is the difference of this line compared to above cudaMemcpy?
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    //// print result, is result 5 instead of 0?
    printf("%d\n", c);

    //// Memory cleanup!
    //// Clean up memory for d_a, d_b, d_c with following example
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
