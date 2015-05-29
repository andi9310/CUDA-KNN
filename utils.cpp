#include "utils.h"
unsigned long long memoryToAllocSize(unsigned long long d, unsigned long long tc, unsigned long long cc, unsigned long long k)
{
	return 4*((1+d)*(tc+cc)+2*cc*k+(MAX_CLASS_NUMBER+tc)*cc);
}
