#include <iostream>
#include "h1.cuh"

__global__ 
void naiveMatrix( const int *a, const int*b, int *c, int N )
{
    // now we address each threads x and  y vals i.e. calculate index for row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //now we will do the calcuation for that result [i][j] from the two matrices
    c[row * N + col] = 0; // set that val to 0

    for (int k=0; k< N; k++) //loop for a thread to handle its row and column 
    {
        c[row * N + col] += a[row* N + k] * b[k*N + col]; // this is doing the calculation for our case
    }
}






