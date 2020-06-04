"use strict";
console.log("\nRemove/add neuron/layer test");

// const args = process.argv.splice(0, 2);
const NeuralNetwork = require("../build/src/src").NNetwork;

const nn = new NeuralNetwork(1, 1);

console.log("Starting");
console.log(nn.layout);

nn.addLayer(1, 5);

console.log("Added Layer");
console.log(nn.layout);

nn.removeLayer(2);

console.log("Removed Layer");
console.log(nn.layout);

nn.layers[1].addNeuron();

console.log("Added Neuron");
console.log(nn.layout);

nn.layers[1].removeNeuron(2);

console.log("Removed Neuron");
console.log(nn.layout);

console.log("Remove/add neuron/layer test finished");