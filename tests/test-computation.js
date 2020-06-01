"use strict";
console.log("\nComputation test");

// const args = process.argv.splice(0, 2);
const NeuralNetwork = require("../build/src").NNetwork;

const nn = new NeuralNetwork(3, 2, 3);

console.log("Weights before:");
console.log(nn.layers.map(v => v.neurons.map(v => Array.from(v.weights))));
console.log("Biases before:");
console.log(nn.layers.map(v => v.neurons.map(v => v.bias)));

nn.train([[5, 34, 3], [74, 7.2, 5.4]]);

console.log("Weights after:");
console.log(nn.layers.map(v => v.neurons.map(v => Array.from(v.weights))));
console.log("Biases after:");
console.log(nn.layers.map(v => v.neurons.map(v => v.bias)));

console.log("Computation test finished");