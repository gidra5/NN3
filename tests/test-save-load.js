"use strict";

// const args = process.argv.splice(0, 2);
const NeuralNetwork = require("../build/src");

let nn = new NeuralNetwork([1, 2, 1]);



nn.saveTo("./temp/nn");

nn = NeuralNetwork.loadFrom("./temp/nn");