## ERLANN - a neural network library for Erlang
======
### What does this do?
- The library automates the creation of the neural network from a user defined number of input perceptrons, number of hidden layers and hidden perceptrons for each layer, and the number of output nodes.
- The library currenty accepts time-series data or a list of data as input data for the network.
- Data is preprocesssed.
- The library trains the network for 80% of the randomized set of training points and tests the network for the remaining 20% as testing points.
- The neural network is evaluatied using the MSPE which is also a stopping criteria for training.

### Specifications
- **Design:** Single perceptron = single Erlang process
- **Data Preprocessing:** Log Normalization
- **Activation Function:** Sigmoid Function
- **Learning Algorithm:** Backpropagation
- **Training Sequence:** Training -> Testing -> Error Evaluation
- **Evaluation Criteria:** Mean Squared Prediction Error (MSPE)

### References
- [Erlang and Neural Networks by Wil Chung](https://erlangcentral.org/wiki/index.php/Erlang_and_Neural_Networks)
- [Fast Artificial Neural Network (FANN) library](http://leenissen.dk/)

### License
GNU GPL v3

### Copyright
2013 MAGNUM TEAM. Mindanao State University - Iligan Institute of Technology.



