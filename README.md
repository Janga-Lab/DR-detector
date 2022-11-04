# DR-detector: A framework for Combining Transfer Learning with Retinal Lesions Features for Accurate Early Detection of Diabetic Retinopathy
The following softwares and modules should be installed before using DR-detector

python 3.6.10

numpy 1.18.1

pandas 1.0.1

sklearn 0.22.2.post1

tensorflow 2.0.0

keras 2.3.1 (using Tensorflow backend)

Running DR-detector:

In order to run DR-detector, the user has do the following:

1- Ensure that the retinal images used for training are placed in one folder called training and the retinal images used for testing are placed in one folder called testing under the main working directory of DR-predictor

2- The CSV file that consists of lesion features for each image in the training dataset as well as the corresponding DR-label is placed in the main working directory of DR-predictor.

3- The CSV file that consists of lesion features for each image in the testing dataset as well as the corresponding DR-label is placed in the main working directory of DR-predictor.

4- Run the following python command:

python main.py 
