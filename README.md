# Learning Pytorch

Welcome to this repository, which documents my comprehensive journey of learning PyTorch from the ground up. Throughout this repository, you will find a variety of projects that I undertook while exploring PyTorch. Each project serves as a practical application of the concepts learned in the corresponding sections, allowing for hands-on experience and deeper understanding. The learning path is organized into 10 distinct sections, each progressively increasing in complexity. This structured approach not only covers the fundamental principles of PyTorch but also dives into more advanced topics. By the end of this journey, the goal is to achieve a mastery-level understanding of PyTorch, equipping myself with the advanced skills needed to tackle any complex machine learning problems.

# Projects Completed

## Linear Regression Model

- **Section**: Workflow Section
- **Description**: Implemented a linear regression model using PyTorch's linear layers on a dummy dataset to predict continuous values.
- **Takeaways**: 
    - Created a basic model class in PyTorch.
    - Explored various loss functions and optimizers.
    - Fundamental structure of model training.
- **Link**: [Here](02-Workflow/workflow.ipynb)

## Classification Model with Non-Linear Activations

- **Section**: Classification Section
- **Description**: Built a classification model using non-linear activation functions on a synthetic dataset generated with `sklearn`'s `make_circles` function.
- **Takeaways**: 
    - Understanding the importance of non-linear activations for complex data patterns.
    - Visualized the difference between using only linear layers and incorporating non-linear functions.
    - Explored various non-linear activation functions and their impact on model performance.
- **Link**: [Here](03-Classification/classification.ipynb)

## FashionMNIST Classification Series

>    A series of projects aimed at building, experimenting with, and improving a classification model on the FashionMNIST dataset. This series explores the impact of different model architectures, from simple linear layers to convolutional neural networks (CNNs). This is the project from the `ComputerVision` section.

- **Name**: FashionMNIST with Linear Layers
    - **Description**: Developed a basic classification model using only linear layers to classify FashionMNIST images.

- **Name**: FashionMNIST with Non-Linear Activations
    - **Description**: Enhanced the classification model by incorporating non-linear activation functions.

- **Name**: FashionMNIST with Convolutional Neural Networks
    - **Description**: Built a CNN-based model for the FashionMNIST dataset, significantly improving accuracy by leveraging convolutional layers.
    - **Takeaways**: 
        - Convolutional layers and Pooling layers.
        - The Non-Linear model was the worst performing model with ~60% accuracy whereas both Linear and Convolutional Networks reached ~80% for three epochs.
    - **Link**: [Here](04-ComputerVision/computer_vision.ipynb)

## Food-3 Classification

> The Food-3 Classification project is a machine learning task focused on classifying three types of food: pizza, steak, and sushi. The model is trained using data extracted from the Food-101 dataset, which contains a large collection of food images. This project demonstrates various approaches to building a classification model, including training a model from scratch and leveraging pre-trained models using transfer learning.

- **Custom Model Training**: In the initial phase of the project, a model was trained from scratch. The training process in `CustomDatasets` faced challenges due to limited data, insufficient model training, and the inability of the model to learn meaningful features from the data. As a result, the model's performance was suboptimal, achieving only ~50% accuracy on the test set.
- **Transfer Learning**: In `TransferLearning` section, the base layers of the EfficientNet model is reused, which was trained on a large and diverse dataset, the model was able to extract relevant features from the food images. Fine-tuning only the `classifier` layer of the model resulted in a significant improvement, with the model reaching around 90% accuracy on the test set.
- **Experiment Tracking**: The project also involved experimenting with different training configurations, including testing various models and learning rates. The `ExperimentTracking` section used TensorBoard for visualizing training metrics, such as loss and accuracy, which helped in monitoring the effectiveness of different approaches and making data-driven adjustments to improve the model's performance.