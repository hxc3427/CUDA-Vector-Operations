# CUDA-Vector-Operations

Vector operations are basic operations that have many applications across a variety of systems and programs, such as Matlab.
While there are some operations that depend on multiple values of a vector, simple operations, such as addition and scaling,
are component-wise operations. Since there are no data dependencies within the vector, these can easily be parallelized.
In this exercise, you will be parallelizing vector addition, subtraction, and scaling, along with gaining experience at
how to write a CUDA kernel. After creating the kernel and doing some analysis, you will be performing an additional
procedure to expose you to the use of page locked memory. By using this type of memory, the explicit memory copies can be 
avoided because the CPU and GPU can access the same memory space.
