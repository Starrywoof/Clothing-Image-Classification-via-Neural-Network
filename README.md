# Neural Network Image Classification Models

In this project, two different types of neural networks, the **Artificial Neural Network (ANN)** and the **Convolutional Neural Network (CNN)**, were trained to classify images of clothing categories. 

## 1. Artificial Neural Network (ANN) Model

The structure of an ANN model can be summarized into three layers: the Input layer, Hidden layer(s), and Output layer. Here is a brief description of each layer:

![Brief Illustration of ANN](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Brief_Illustration_of_ANN.png)

**Figure 1:** Brief Illustration of ANN

The Input layer consists of training set images, and the Output layer consists of 10 clothing category labels. 

A *Flatten layer* is employed to convert the multidimensional 28x28 image data into a one-dimensional format. Consequently, when a set of 28x28 image data is input and passed through the Flatten layer, we obtain 784 neurons.

The connection between the Hidden layer and the Input layer is fully connected, so we use the Dense layer from Keras.

![An Example of Dense (ReLU) Layer](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Fully_Connected.png)

**Figure 2:** An Example of Dense (ReLU) Layer

In classification problems, Softmax Regression is commonly used to calculate the probabilities of output classes.

![The Dense (Softmax) Regression Layer](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Regression_Layer.png)

**Figure 3:** The Dense (Softmax) Regression Layer

### Visualization through Tensorboard

![Training Accuracy Curve and Loss Curve](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Accuracy_and_Loss_Curve_ANN.png)

**Figure 4:** Training Accuracy Curve and Loss Curve

![Histograms and Distributions Graph](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Histograms_and_Distributions_ANN.png)

**Figure 5:** Histograms and Distributions Graph

<small>**Note:** The data source for Histograms graphs consists of multiple one-dimensional arrays. In Distributions graphs, the x-axis corresponds to the array ID, while the y-axis corresponds to the color values in the array. The darkness or lightness of the colors indicates the frequency of occurrence for the corresponding values.</small>

### Visualizing Clustering Process using Projector

![Initial Clustering Situation of the Softmax Layer](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Initial_Clustering_ANN.png)

**Figure 6:** Initial Clustering Situation of the Softmax Layer

![Clustering Situation of the Softmax Layer at Iteration 279](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Clustering_Iteration_ANN.png)

**Figure 7:** Clustering Situation of the Softmax Layer at Iteration 279

## 2. Convolutional Neural Network (CNN) Model

![Convolutional Neural Network Model Architecture Diagram](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/CNN.png)

**Figure 8:** Convolutional Neural Network Model Architecture Diagram

![An Example of Convolution Operation](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Convolution_Operation.png)

**Figure 9:** An Example of Convolution Operation

![An Example of Max Pooling](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Max_Pooling.png)

**Figure 10:** An Example of Max Pooling

### Visualization through Tensorboard

![Training Accuracy Curve and Loss Curve](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Accuracy_and_Loss_Curve_CNN.png)

**Figure 11:** Training Accuracy Curve and Loss Curve

![Histograms and Distributions Graph](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Histograms_and_Distributions_CNN.png)

**Figure 12:** Histograms and Distributions Graph

![Visualizing the Feature Maps of a CNN Model through Forward Computation](https://github.com/Starrywoof/Clothing-Image-Classification-via-Neural-Network/blob/main/Pictures/Feature_Maps_CNN.png)

**Figure 13:** Visualizing the Feature Maps through Forward Computation

## 3. Conclusion:

In the image classification task of 10 different clothing categories, the **ANN model achieved an accuracy of over 88%**, while the **CNN model achieved an accuracy of over 91%**. The training logs of the neural networks were recorded using Tensorboard tool. The feature maps of the convolutional layers in the CNN model were visualized by performing forward computations.

## 4. Data Source

The data for this project is sourced from the Fashion-MNIST dataset on Kaggle. It primarily utilizes four compressed files: "t10k-images-idx3-ubyte," "t10k-labels-idx1-ubyte," "train-images-idx3-ubyte," and "train-labels-idx1-ubyte."

The extraction of files is primarily done using the gzip.open() function in Python. The content of the files is read using the functionalities of the numpy library.

To normalize the pixel data of the images, which are stored as train_images and test_images datasets, you need to divide them by 255. This will normalize the pixel values to the range of 0 to 1, suitable for training a neural network model.
