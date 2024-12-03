# Stock_Transformer
# This is just a transformer that I modified from a language translation transformer
# I changed it to encode using Time2Vec instead of positional encoding, and removed the decoder since it's not "generating" words, and it seemed unnecessary when it's just producing a single output each time. 
# The actual model is trained with train.py, train2.0 was just an experiment. The model that predicts 10 minutes out is model.py, and the model that just predicts the next minute is model_one.py. 
