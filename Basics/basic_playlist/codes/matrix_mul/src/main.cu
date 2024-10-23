#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <vector>


#include "h1.cuh"


//func to do naive matrix multiplication on CPU to check against the already obtained result

void seqMatrix( std :: vector<int> &a, std::vector<int> &b, std::vector<int> &c, int N )
{
    //for each row
    for (int i=0; i<N;i++)
    {//for each column
        for(int j=0;j<N;j++)
        {
            int temp = 0;
            for(int k=0; k<N; k++)
            {
                temp+= a[i * N + k] * b[k * N + j]; // this is doing the row and col calculations partially
            }
            assert(temp == c[i*N + j]);
        }
    }
}


int main()
{


std::cout <<"the naive parallel algo :\n";

// a matrix sizes of 2^10 * 2^10

int N = 1 << 10;

// taking the size of matrix to give to device for launching threads

size_t one_matrix = N * N * sizeof(int);

// cpu vectors
std::vector<int> h_a(N*N);
std::vector<int> h_b(N*N);
std::vector<int> h_c(N*N);

// get random numbers filled in the matrices

std::generate(h_a.begin(), h_a.end(),[](){ return rand() % 100; });
std::generate(h_b.begin(), h_b.end(),[](){ return rand() % 100; });

//std::cout <<"the matrics look like this :\n";
// for(int r = 0; r < N; r++)
// {
//     for ( int c = 0; c < N; c++ )
//     {
//         std::cout <<" "<<h_a[r * N + c]<<" ";
//     }
//     std::cout<<"\n";
// }

// allocating the device memory for the data for calculation
int * d_a, *d_b, *d_c;
cudaMalloc(&d_a, one_matrix);
cudaMalloc(&d_b, one_matrix);
cudaMalloc(&d_c, one_matrix);

//transfer
cudaMemcpy( d_a, h_a.data(), one_matrix, cudaMemcpyHostToDevice );
cudaMemcpy( d_b, h_b.data(), one_matrix, cudaMemcpyHostToDevice );

int THREADS = 32; // set the number of threads we want to launch in one block

int BLOCKS = N / THREADS; // set the number of blocks always as a multiple of 32 (Architecture quirk)

dim3 onethread ( THREADS, THREADS ); // using dim3 data struct to store the dimensions of the threads and the blocks we want on the device (gpu)
dim3 oneblock ( BLOCKS, BLOCKS );

//dimensions:
// one thread : 32 x 32 x 1 
// one block : v x v x 1 where v = N / 32 = 1024 / 32 = 32 only

// now to launch the kernel:
naiveMatrix<<< oneblock, onethread >>> (d_a, d_b, d_c, N); // tell device the dimensions of the blocks and threads we want use.

//the kernel code includes the checks against index for correct calculations

//transfer
cudaMemcpy(h_c.data(), d_c, one_matrix, cudaMemcpyDeviceToHost);

seqMatrix(h_a, h_b, h_c, N); // this calculates results  on cpu and the assert() checks if the result is same from both the places.

std::cout<<"done !\n";

// dont forget to free the memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);




return 0;

}
