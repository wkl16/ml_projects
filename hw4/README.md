Download tar.gz file from this website  
https://ai.stanford.edu/%7Eamaas/data/sentiment/  
Unzip in the hw4 dir.  
  
Run generate_data.py, and see results.  
The function process_data() processes the data from the acmlImdb directory,  
and runs it through a very basic FNN (accuracy is really bad)  

- Note from Matt -
Not sure if data is properly being preprocessed, but figured
not to go any further in the tuning of parameters
outside of what is present in order for us to see more tangible
results of training. Potential issues could be of the following:
- The data is not being preprocessed correctly
- The model is too simple
- Using the wrong type of loss function
- testing and training are not done in correct manner (though I based it off of
  in class demonstration of FNNs)
- Too few features from tfidf Vectorization

- End note from Matt -
