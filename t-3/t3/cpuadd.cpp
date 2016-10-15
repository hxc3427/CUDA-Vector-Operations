#include "common.h"

void addVectorCPU( float* a, float* b, float* c, int size ){
	for (int i=0; i < size; i++)
               {
                      c[i] = a[i] + b[i];
               }
}