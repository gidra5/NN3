# Usage examples

## XOR problem

First we can define training data, that we will use:

    const trainingData = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]];
    
> In general data may be loaded from file, not hardcoded, so be sure to follow this format of nested arrays.

Construct NN specifying its layout (amount of neurons per layer in order).  
> First layer will be input layer, so it should match dimensions of expected input (2 inputs).  
> Last layer will be output layer, so it should match dimensions of expected output (1 output).  
> Layout is motivated by this [post](https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b).  

    const nn = new NeuralNetwork(2, 2, 1);

Default activation function (sigmoid) is apropriate for this problem, so no change of it is needed.  
Now train NN with dataset defined previously:  

    nn.train(...trainingData);

Use trained NN to calculate XOR of some values.  
Trivial case from dataset:  

    nn.feedforward(1, 0);

Non-trivial case to see generalization:  

    nn.feedforward(0.5, 0.7):

Save trained NN for future use:  

    nn.saveTo(`${process.cwd()}/trained/example-1.nn`);

For full source code of this example check [`examples/example-1.ts`](https://github.com/gidra5/NN3/blob/master/examples/example-1.ts)
