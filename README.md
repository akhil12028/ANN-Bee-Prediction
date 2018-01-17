# ANN-Bee-Prediction
AN artificial neural network model to classify the images as bees or no bees.


load_data_conv.py - loading the data and storing it in the pickle file output.dat.

final_project.py - training the neural network with the given data and saving the model in bees_final3.model.

final_model.py - implementation of the testNet(netpath,dirpath) function.

-> To run the code on a directory of images or audio files, execute the final_model.py file.

Before, executing change the following variables

netpath - path of "bees_final3.model" file

dirpath - directory of the images to be passed

After execution, it prints out the predicted classification of the images in the directory. 
