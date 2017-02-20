# Polyglot ML

The rather ambitious goal of this project is to implement machine learning
techniques in a suite of different languages with the primary purpose of
helping the user understand what the model is doing in the training process.

Using the content from the [Coursera Machine Learning course][coursera_ml]
taught by Andrew Ng and the [Stanford Deep Learning Course][stanford_ml] as the
reference implementations, we'll start with Octave, Python, and OCaml as the
first three languages that this library supports.

Want to implement the following:

- Linear Regression
- Logistic Regression
- K-means Clustering
- PCA
- Feed Forward Neural Nets
- Support Vector Machines
- Softmax Classifier
- RICA
- ZCA
- Convolutional Neural Nets

The goal is to stay fairly low level and use the language's most stable linear
algebra libraries to implement each model. As needed, we'll also probably use
pre-existing packages and higher-level functions for things like optimization,
just so user can focus on the core training logic of each model type.

This project is still under construction and not ready for use :)

[coursera_ml]: https://www.coursera.org/learn/machine-learning
[stanford_ml]: https://github.com/amaas/stanford_dl_ex
