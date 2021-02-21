
Fill the current report and commit it as any other file (be compact, a few lines per Section should be enough).

# TO INSTALL PERF ON WSL 2
```
# WARNING: most features of perf will not work because WSL doesnt support hardware counters
# See the issue https://github.com/microsoft/WSL/issues/4678
apt install flex bison
git clone https://github.com/microsoft/WSL2-Linux-Kernel --depth 1
cd WSL2-Linux-Kernel/tools/perf
make -j8
sudo cp perf /usr/local/bin
```

RUN THE CODE WITH THE LINE:
```
OMP_NUM_THREADS=8 OMP_PROC_BIND=TRUE perf stat -B build/matrix -no-check
```

RUN PERF ON ATLAS WITH
```
perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations,L1-dcache-load-misses,LLC-load-misses,LLC-store-misses build/matrix -no-check
```

# At the end of the practical work

## What worked

## What did not work



# Final status

## What works

## What does not work

