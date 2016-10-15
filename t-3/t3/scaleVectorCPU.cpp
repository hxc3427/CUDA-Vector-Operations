#include "common.h"

void scaleVectorCPU( float* a, float* c, float scaleFactor, int size ){
	for (int i=0; i < size; i++)
               {
                      c[i] = scaleFactor * a[i];
               }
}