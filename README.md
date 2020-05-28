#Concept

Just another NN-learning library for personal usage and education

#Installment

After cloning you can already import this lib (`src/src.ts` file) to your TS project, but to import to JS project you need to build TS code
`npm install` will install devDependencies and compile TS code to JS code into `/build` folder

#Tests

Tests can be run with `npm test` (after `npm install` was once called) which will lint and run test scripts from `/tests` folder
Tests check wheather code works for simplest case of NNs.

#Usage

After importing lib as NN you can:

- create NN with specific layout like this:

        const nn = new NN([1, 1]); //makes NN with 2 layers, each with 1 neuron

- get output of NN via feedforward():

        const out = nn.feedforward(data);

    Note that input data should match first layer's number of neurons and output will match last layer's number of neurons, or exception will be thrown
- learn input-output pair with backprop():

        nn.feedforward(input); //will set apropriate values for neurons
        nn.backprop(output);   //learn with backpropagation

- import/export NN from file with `saveTo()` and `loadFrom()`
