"""
Synthesis for each rela through each pair

Func to take in original syn (e.g. HH or Main) and pool to output the second syn
This will output the which rec cannot be found in the pools as well (rare combinations)
Also a wrapper to take in all pools and the original (only HH)
Then we will combine all again and output the kept and removed 
This can go through SAA again (with the help of the loop check census)
"""
