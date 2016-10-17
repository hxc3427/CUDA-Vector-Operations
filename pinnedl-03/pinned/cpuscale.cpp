//
//  CPU.cpp
//  hpalab5
//
//  Created by Harshdeep Singh Chawla on 10/4/16.
//  Copyright © 2016 Harshdeep Singh Chawla. All rights reserved.
//
#include "common.h"

void scaleVectorCPU( float* a, float* c, float scaleFactor, int size ){
	for (int i=0; i < size; i++)
               {
                      c[i] = scaleFactor * a[i];
               }
}
