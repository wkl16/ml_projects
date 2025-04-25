Download tar.gz file from this website  
https://ai.stanford.edu/%7Eamaas/data/sentiment/  
Unzip in the hw4 dir.  

pip install these in addition to requirements.txt  
in the root directory of this repository:  
- pyprind (though can comment out, this one is unnecessary)
- torch
- nltk
  
Run generate_data.py, and see results. (Task 2)  
The function process_data() processes the data from the acmlImdb directory,  
and runs it through a FNN similar to the one seen in lecture.  
generate_data.py was modified by Leo Wang to ensure the correct  
splitting of data, and providing more metrics to compare the performance  
of the FNN with. 