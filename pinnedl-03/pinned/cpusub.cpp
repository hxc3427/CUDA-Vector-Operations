//
//  CPU.cpp
//  hpalab5
//
//  Created by Harshdeep Singh Chawla on 10/4/16.
//  Copyright © 2016 Harshdeep Singh Chawla. All rights reserved.
//
#include "common.h"

void subtractVectorCPU( float* a, float* b, float* c, int size ){
	for (int i=0; i < size; i++)
               {
                      c[i] = a[i] - b[i];
               }
}
