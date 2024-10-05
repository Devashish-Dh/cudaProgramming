#include <iostream>

__global__ // device code i.e. the "kernel" that will run on the GPU
void vecAddKernel ( float* A, float* B, float* C, int n )
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

// the func that adds the vectors traditionally with a loop on the CPU :
void vecAdd ( float* A_h, float* B_h, float* C_h, int n )
{
    for ( int i = 0; i < n; i++ ){
        C_h[i] = A_h[i] + B_h[i];
    }
}

// host code
void deviceVecAdd ( float* A_h, float* B_h, float* C_h, int n )
{
    int size = n * sizeof( float ); // calculating the memory space needed on the device for our vectors

    float *A_d, *B_d, *C_d; //memory addresses of device (CPU cannot de-reference them)

    cudaMalloc( (void**) &A_d, size); // tells device to allocate "size" amount of space and address it using the "A_d"
    cudaMalloc( (void**) &B_d, size); //vector B

    cudaMalloc( (void**) &C_d, size); // output will be stored here on the device

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice); // to send the vectors FROM cpu TO device
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // call to the kernel on the device BY THE cpu
    vecAddKernel <<< ceil(n/256.0), 256 >>>(A_d, B_d, C_d, n); // this has the <<< >>> that tells the device how many threads per block and how many blocks to use 

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost); // getting the calculated results FROM device TO host

    // very important to free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}



int main()
{
int n = 5; 
// input vectors
float a[n] = {1,2,3,4,5};
float b[n] = {6,7,8,9,10};

// vector to hold results of the sum
float c[n];
float c_from_d[n];

vecAdd(a, b, c, n); // cpu does this

std::cout<<" CPU summed C["<<n<<"] : ";
for (int i=0; i<n; i++){
    std::cout<<" "<<c[i]<<" ";
}
std::cout<<"\n";



deviceVecAdd(a, b, c_from_d, n); //gpu does this 

std::cout<<" GPU summed C_from_d["<<n<<"] : ";
for (int i=0; i<n; i++){
    std::cout<<" "<<c_from_d[i]<<" ";
}
std::cout<<"\n";



return 0;
}
