"use strict";

// const args = process.argv.splice(0, 2);
const NeuralNetwork = require("../build/src");

const nn = new NeuralNetwork([1, 2, 1]);

nn.feedforward([5]);
nn.backprop([67]);