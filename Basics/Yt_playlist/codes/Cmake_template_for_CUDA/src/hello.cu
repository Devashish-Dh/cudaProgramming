#include <iostream>
#include "h1.cuh"

//__global__ void myKernel(void) {}

int main()
{

std::cout << "hello cuda and C / C++ together ! \n";

myHelloKernel <<< 1, 1 >>>();

std::cout << "Hello CUDA!\n";

return 0;

}
