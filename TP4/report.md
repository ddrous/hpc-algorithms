
Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work

## What worked
7. **`dot`:** the unaligned version worked (but I'm not sure of the result)
## What did not work
8. **`matmat`:** I had no time to implement this!

</br>

# Final status

## What works
7. **`dot`:** everything works as expected. The AVX2 dot product often offers the best performance.  
8. **`matmat`:** works as expected. The AVX2 matrix product is often twice as fast. For the function `matmat4x4_avx` to accurately perform the matrix product `C=A*B`, two assumptions are made when passing `B` and `C` as arguments:  
    - **`B` must be transposed** beforehand for the program to perform multiplications row by row 
    - **`C` must be empty**, as the program doesn't overwrite its data

## What does not work
Nothing doesn't work.

