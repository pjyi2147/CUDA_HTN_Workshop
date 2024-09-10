#include <stdio.h>

__global__ void example_kernel() {
    printf("this is an example!");
}

int main() {
    example_kernel<<<1,1>>>();
    return 0;
}
