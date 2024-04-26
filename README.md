# MALARIA DETECTION USING NEURAL NETWORKS

Malaria is a parasitic disease transmitted to humans through the bite of infected female Anopheles mosquitoes of the Plasmodium group. There are five main parasite species responsible for transmitting malaria infection: Plasmodium falciparum, Plasmodium vivax, Plasmodium ovale, Plasmodium malariae, and Plasmodium knowlesi. While malaria can affect anyone, certain demographics face a higher risk and burden of the disease. Pregnant women and children under five are particularly vulnerable. Young children lack developed immunity against severe forms of malaria, making them more susceptible to its adverse effects. Pregnant women are also at increased risk, as malaria infection during pregnancy can lead to complications such as miscarriage, low birth weight, and maternal and newborn mortality.
Malaria remains one of the most prevalent and life-threatening infectious diseases globally, particularly in regions with limited access to healthcare resources. The conventional methods for diagnosing malaria, such as microscopic examination of blood smears, can be time-consuming, labour-intensive, and prone to human error. Therefore, there is a critical need for automated and accurate diagnostic tools to assist healthcare professionals in timely and reliable malaria detection.



## Approach

The neural network model architecture employed for malaria detection is based on Convolutional Neural Networks (CNNs). CNNs are well-suited for image classification tasks due to their ability to learn hierarchical features directly from raw pixel data. The model is designed to learn discriminative features from cell images and classify them into infected or uninfected classes. To enhance the performance and accuracy of the CNN model, we plan to conduct hyperparameter tuning. Hyperparameter tuning involves optimizing key parameters that significantly impact the model's learning and generalization ability for better classification.

# Process Overview

#### Image Preprocessing:
• Images are rescaled to match the available computational resources while preserving essential features for accurate classification.
#### Training the Convolutional Neural Network Model:
• The initial step involves training the Convolutional Neural Network (CNN) model on the preprocessed images. The trained model is saved for further evaluation and comparison.
#### Initial Accuracy Testing:
• The trained CNN model is tested for accuracy using a separate test dataset to assess its performance.
#### Hyperparameter Tuning with GridSearch:
- Identifying Key Hyperparameters: We will identify and prioritize hyperparameters such as batch size, nodes, layers and activation function in each layer.
- Hyperparameter Search and Evaluation: Using the chosen parameters, we train multiple instances of the CNN model with different hyperparameter combinations. Each configuration will be evaluated based on performance metrics such as accuracy on a test set.
#### Model Selection: 
• Based on the accuracy for the test data from all the configurations, the one with the best accuracy is choosen.
#### Misclassified Images:
• Misclassified images from the tuned CNN model are analyzed to understand the model's limitations and areas for improvement.
#### Comparative Analysis with ResNet:
•	In addition to the CNN model, a ResNet model is trained and saved to evaluate its usability and performance compared to CNNs. This comparative analysis provides insights into the effectiveness of different neural network architectures for malaria detection.
#### Evaluation and Validation:
• Both the CNN and ResNet models are evaluated using standard metrics such as accuracy on a test dataset to validate their performance and assess their suitability for real-world applications.
 
By following this structured process, we aim to develop robust and accurate neural network models for malaria detection, leveraging hyperparameter tuning and comparative analysis to achieve optimal performance and reliability.



# EDA
[Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

#### Dataset Description
This dataset comprises a collection of blood smear images captured from malaria-infected and uninfected patients. The images were obtained using a light microscope at varying magnifications and contain red blood cells infected with different species of the Plasmodium parasite, including Plasmodium falciparum and Plasmodium vivax, among others.

The dataset is organized into two main categories:       

1 -> Parasitized Cells: Images of red blood cells infected with malaria parasites.

0 -> Uninfected Cells: Images of healthy red blood cells without any malaria infection

![download](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/20a7402c-7d29-410a-a95a-7a6b57aa318d)


X: Cell images.    
y: 0/1

Problem type: Classification      

Each category contains 13780 high-resolution images in JPEG format, providing a diverse and comprehensive dataset for training and evaluating machine learning models for malaria detection. The Shape of each of the individual image is: (148, 142, 3)

![infected](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/59b33d41-6f7c-4924-9a39-8b8624fe67ac)
![uninfected](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/55c43364-f303-4be2-a350-70d3eec15d8a)





## Model fitting
- Having generous number of samples we change 9:1 split for train and test. And another 10% for validation from training samples. Having generous amount of samples we wanted the model to detect more patterns for prediction.

- Scaling down the images to 64x64 did not have a significant impact on accuracy values making no data leakage.

- CNN are best suited of image classification, the data does not have much variations in orientation or spatial arrangments. CNN being invariant was the best choice for this work.

- GridSearch method was used to select the hyperparamters

## Validation / metrics

- Accuracy provides a comprehensive measure of a model's overall performance across all classes. It gives a single numerical value that summarizes the correctness of predictions without considering class-specific performance

Corectly Classified Samples:
![True pred](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/882e698e-ccbc-4ab9-8ba1-f716b8614ac4)

Misclassified Samples:
![False pred](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/91faa030-861c-482c-998e-e75aa88d232d)


![Confusion matrix](https://github.com/nipun-davasam/IA651-Applied-Machine-Learning/assets/151178533/ab34d377-e7b1-4789-b9d9-65c11f72cb57)


## Production

Mobile applications for remote diagnosis: Mobile apps equipped with malaria detection models can enable individuals in remote or rural areas to perform self-testing for malaria using their smartphones. The app can capture images of blood smears or use rapid diagnostic tests (RDTs) to provide preliminary diagnoses, which can then be verified by healthcare professionals.


## Going further
Need to test on noisy images. We are planning to change the threshhold of the predictions and try to reduce the type 1 error(False-Positive) cases among the predictions.

