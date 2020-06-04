# Concept

Just another NN-learning library for personal usage and education

# Usage

## Importing

*On Linux* after cloning you can already import this lib (`src/src.ts` file) to your TS project, but to import to JS project you need to build TS code.
`npm install` will install devDependencies and compile TS code (example and src) into `/build` folder

## Functionality

After importing lib as NN you can:

- create NN with specific layout like this:

        const nn = new NN([1, 2, 1]); //makes NN with 3 layers with 1, 2 and 1 neuron respectively

- get output of NN via `feedforward()`:

        const out = nn.feedforward(data);

    Note that input data should match first layer's size or exception will be thrown
- learn input-output pair with `train()`:

        nn.train([input, output]);

    Note that dimensions of input and output should match first and last layer's size respectively

- dynamically change NN's structure with `addNeuron()`/`addLayer()` and `removeNeuron()`/`removeLayer()` functions

- import/export NN from file with `saveTo()` and `loadFrom()`

Other available function is described in [`Docs.md`](https://github.com/gidra5/NN3/blob/master/Docs.md)

# Tests

Tests can be run with `npm test` (after `npm install` was once called) which will lint and run test scripts from `/tests` folder
Tests check wheather code works for simplest case of NNs and some features.

# Examples

Examples of usage are available in [`Examples.md`](https://github.com/gidra5/NN3/blob/master/Examples.md)  
You can run sources after building project with `node ./build/*example*`
