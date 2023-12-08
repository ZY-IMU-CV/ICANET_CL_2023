# ICANET_CL_2023
The official implementation of “ICANet: A LIGHTWEIGHT INCREASING CONTEXT AIDED NETWORK FOR REAL-TIME IMAGE SEMANTIC SEGMENTATION”

ICANet
 
Decoder
 
Setup
Install the dependencies in requirements.txt by using pip and virtualenv.

Download Cityscapes
Go to https://www.cityscapes-dataset.com, create an account, and download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip. Unzip both of them and put them in a directory called cityscapes_dataset. The cityscapes_dataset directory should be inside the ICANet directory. If you put the dataset somewhere else, you can set the config field.

Usage
To see the model definitions and do some speed tests, go to model.py.
To train, validate, benchmark, and save the results of your model, go to train.py.

Cityscapes results
Test mIoU: 75.3%
Val mIoU: 75.7%


