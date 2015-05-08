#include <iostream>


__global__ void classify(int * teached)
{

}



int main()
{

	classify<<<100,100>>>(NULL);
	std::cout << "ok"; 


}
