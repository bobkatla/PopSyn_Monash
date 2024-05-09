# Population Synthesis

Various implementation including test for Population Synthesis.

Will go through a big refactoring now!

TODO: unit testing
Also consider using click to use the CLI

Especially need to organise the methods to have them clean well, seperate into different files.

Proposed structure:

- DataProcessor: contain all raw data, processed data. Should make this the unified one. Maybe make it a class and then we can generate data to init the data folder for each method
- Utils: This will include all common utils. First is Benchmark to help calculate all needed scores. Need to make it generalised so later we can reduce repetitive code. Next is the rounding (TRS), can implement more. Also any extra will think and put here
- Method: each method will be inside a folder, it will have it own data and output folder to seperarate each of them. The DataGenerator should help init data folder for all of them
- Notebooks: we will put it outside for further analysis and get results. Especially this is where we get the visualisation and compare methods.
- Tests: we will do this last, this will contain test for all needed one. Need to have pipeline for unit test as well. 