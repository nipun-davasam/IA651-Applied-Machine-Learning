
## Approach

The neural network model architecture employed for malaria detection is based on Convolutional Neural Networks (CNNs). CNNs are well-suited for image classification tasks due to their ability to learn hierarchical features directly from raw pixel data. The model is designed to learn discriminative features from cell images and classify them into infected or uninfected classes

## Process

- Rescale the images to available computational resources
- Train the neural model and save it
- Test for accuracy
- Repeat this process using GridSearch method to get best paramters for the model
- Observe the miss classified images from the model save and handle it 
- Train ResNet model and save it to compare its usability in comparision to CNNs

## EDA
We have collected the dataset from Kaggle available at the following link: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

This dataset comprises a collection of blood smear images captured from malaria-infected and uninfected patients. The images were obtained using a light microscope at varying magnifications and contain red blood cells infected with different species of the Plasmodium parasite, including Plasmodium falciparum and Plasmodium vivax, among others

X: Cell images.    
y: infected and uninfected 

Problem type: Classification        

## Dataset Description
The dataset is organized into two main categories:       
** Parasitized Cells** : Images of red blood cells infected with malaria parasites.

Uninfected Cells: Images of healthy red blood cells without any malaria infecti

