"""This should be easier
with each connection we build a model
each model will have the ability to sample from the knowing (we can setup to say all possible values for each att)
we can assume all possible states from seed
then we build the models
then we sample given evidence for each att
So to do that we need to do groupby to have the syn_count
BN will be just forward sampling not backward to update (not needed)
turn out the key is simply adding the count into sampling
"""