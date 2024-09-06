#include <stdio.h>

// 7 x 7 Matrix
#define N_X 7
#define N_Y 7

// 4 x 4 blocksize
#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 4

void random_ints(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 10;
    }
}

__global__ void add(int *a, int *b, int *c, int len_x, int len_y)
{
    // Calculate yidx and xidx yourself. Use blockIdx, blockDim, and threadIdx
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;

    // What needs to be checked here?
    if (yidx < len_y && xidx < len_x)
    {
        // What is the index we should use here?
        c[yidx * (len_x) + xidx] = a[yidx * (len_x) + xidx] + b[yidx * (len_x) + xidx];
    }
}

int main()
{
    int size = sizeof(int) * N_X * N_Y;
    int *a = (int *)malloc(size);
    random_ints(a, N_X * N_Y);
    int *b = (int *)malloc(size);
    random_ints(b, N_X * N_Y);
    int *c = (int *)malloc(size);
    memset(c, 0, size);

    // device copies of a, b, c
    int *d_a, *d_b, *d_c;

    //// Allocate memory to d_a, d_b and d_c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);


    //// Copy input to d_a, d_b
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    //// Launch kernel
    //// what is the difference here? We use dim3 struct (x, y, z)
    dim3 NUMBLOCKS = {(N_X - 1) / BLOCK_SIZE_X + 1, (N_Y - 1) / BLOCK_SIZE_Y + 1 ,1};
    add<<<NUMBLOCKS, {BLOCK_SIZE_X, BLOCK_SIZE_Y, 1}>>>(d_a, d_b, d_c, N_X, N_Y);
    cudaDeviceSynchronize();

    //// Copy result from device (d_c) to host (c)
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    //// print result, is result correct?
    for (int i = 0; i < N_Y; i++)
    {
        for (int j = 0; j < N_X; j++)
        {
            printf("i = %d, j = %d, %d + %d = %d\n", i, j, a[i*N_X + j], b[i*N_X + j], c[i*N_X + j]);
        }
    }

    //// Memory cleanup!
    //// Clean up memory for d_a, d_b, d_c
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //// Of course the same for host memory, too.
    free(a);
    free(b);
    free(c);

    return 0;
}
