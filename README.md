# Learning Pytorch

Welcome to this repository, which documents my comprehensive journey of learning PyTorch from the ground up. Throughout this repository, you will find a variety of projects that I undertook while exploring PyTorch. Each project serves as a practical application of the concepts learned in the corresponding sections, allowing for hands-on experience and deeper understanding. The learning path is organized into 10 distinct sections, each progressively increasing in complexity. This structured approach not only covers the fundamental principles of PyTorch but also dives into more advanced topics. By the end of this journey, the goal is to achieve a mastery-level understanding of PyTorch, equipping myself with the advanced skills needed to tackle machine learning projects.

# Projects Completed

## Workflow Section

- **Name**: Linear Regression Model
    - **Description**: Implemented a linear regression model using PyTorch's linear layers on a dummy dataset to predict continuous values.
    - **Takeaways**: 
        - Created a basic model class in PyTorch.
        - Explored various loss functions and optimizers.
        - Fundamental structure of model training.
    - **Link**: [Here](02-Workflow/workflow.ipynb)

## Classification Section

- **Name**: Classification Model with Non-Linear Activations
    - **Description**: Built a classification model using non-linear activation functions on a synthetic dataset generated with `sklearn`'s `make_circles` function.
    - **Takeaways**: 
        - Understanding the importance of non-linear activations for complex data patterns.
        - Visualized the difference between using only linear layers and incorporating non-linear functions.
        - Explored various non-linear activation functions and their impact on model performance.
    - **Link**: [Here](03-Classification/classification.ipynb)

## Computer Vision Section

### FashionMNIST Classification Series
>    A series of projects aimed at building, experimenting with, and improving a classification model on the FashionMNIST dataset. This series explores the impact of different model architectures, from simple linear layers to convolutional neural networks (CNNs).

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