# pharmahacks_multi_disease_classification
### By Rayaq Siddiqui

To run the project simply run:

python main.py

in the console in the main folder.

## Additional Parser Arguments

python main.py --epochs ##some_number## --verbose ##0,1,2## --model ##list_below##

The possible models that you can run are:

1. simple_branchy_linear_network
2. branchy_linear_network
3. seq_model
4. deep_linear_network
5. dual_input_model

The best accuracy is the simple_branchy_linear_network model. To run this model you can run

python main.py --epochs 12 --verbose 1 --model simple_branchy_linear_network

OR 

python main.py

as these are the Defaults.

## Note

All of my research and brainstorming is done in the Jupyter Notebooks, so please feel free to check them out. 

I test all of the models and try to optimize them to the best of my abilities. 

Additionally, apart from neural networks, I test out other Classification methods (i.e. Support Vector Machines, K-Nearest Neighbors, Decision Trees). However, I decided to disclude them since they provided me with a lower accuracy compared to my models.
