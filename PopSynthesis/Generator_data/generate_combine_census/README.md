# Explain the process

1. Create the meta file to know how should we map to match from sample to census
2. Generally, it will map based on name, but there are cases of SHIFT, BREAK, and DELETE
- SHIFT: the census value will be shift to the other columns (not DELETE or SHIFT) equally
- BREAK: it will be followed by the names that we want it to break into (equally)
- DELETE: just delete the value
3. We have a process later to match the data to the expected values, generally speaking the above process is to help maintain the distributions.