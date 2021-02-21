
Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work

## What worked
- The matrix distance

## What did not work
- Nothing else


# Final status

## What works
- The matrix __distance__ works
- The matrix-matrix product __matmat__ works
- The sum of all values in an array __sumvalues__ works: some of the ideas are the following:
    1. Use the maximum number of threads if the size of the array is larger than twice the max number of threads. As each thread performs approximatively `arraySize/(MaxGroup*MaxThreadsPerGroup) - 1` additions, we end up with `MaxGroup*MaxThreadsPerGroup` partial sums.
    2. If the array is small enough, we use a convenient amount of threads: slightly larger (if not equal) to the array size. Each thread performs approximatively `1` addition.
    3. The array size at one step must equal the number of threads used in the previous step (`arraySize = threadNb*blockNb`);
    4. The priority is to reduce the number of blocks (instead of reducing the number of threads per block) as communications in global device memory are costlier than communications in shared device memory.
    5. The buffer `cuBuffer` has to be swapped with (or copied into) `cuValues` between each iteration. We could avoid this by using __interleaved addressing__, or __sequential addressing__, hence modifying the values of the array in-place as we add them ([Nvidia Dev pp.7,14](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)). 

## What does not work
- Nothing doesn't work

