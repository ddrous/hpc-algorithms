
Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# At the end of the practical work

## What worked
- 5 Regex (variables declaration)  
The regular expression I have provided has been tested, and can detect any C++ `integer` declaration (and their optional instantiation) the likes of:
```
    - int a;  
    - int a, b;    
    - int a, b =  3;  
```
The same goes for `double` and `string` (or `std::string`). 

## What did not work
- 5 Regex (decimal numbers and multiline comments)  
- 6 Optimization  
- 7 Backend  


<br/>

# Final status


## What works
- 5 Regex (everything including, decimal numbers and multiline comments)
    - The regular expression has been extended to match scientific notations for `double`.
    - As for multiline comments, the algorithm is as follows:
        1. match the opening sequence `/*` or `/**`.  
        2. match any block that starts with or without `*` or `/`, and ends before `*` or `/`. Note that the beginning `*` or `/` is part of the block while the ending `*` or `/` is not.  
        3. repeat step 2. at least once.  
        4. match the closing sequence `*/`. 
   
- 6 Optimization  
    - Constant propagation: done using a <map> as suggested
    - Remove unused variables: done by starting from the returned value (the final instruction) as suggested

- 7 Backend  
    I had two major optimizations:
    - DECREASE THE NUMBER OF ALLOCATIONS: This is done by avoiding to re-allocate registers for variables that are 'live' (_already contained in a register_). Before allocating any register for a variable, let's check first if that variable has been loaded into/from a register in the past. To do this, we go through the instructions that have been executed up to this point, stopping at the first appearance of our variable. If the variable has been used in the past, we just need to make sure that the corresponding register still contains its value. If the register has been used to store another variable since then, we are not in luck, and we need to reload the variable in a register as if it was the first time we encountered said variable.
    - EFFICIENTLY INCREASE THE NUMBER OF REGISTERS USED: The "dumb algorithm" provided only uses 2 of the available registers (at most). Assuming we have 3 registers available, we could also allocate registers for the second operand. That is what we did. Even though our algorithm considers the second operand, it takes as much instructions(time) as the 'dumb algorithm' (20 instructions) when tested on the set of instructions provided. This is all thanks to 1. that avoids useless re-allocations.


## What does not work
Nothing doesn't work.

