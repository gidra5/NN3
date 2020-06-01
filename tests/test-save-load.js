"use strict";
console.log("\nSave-load test");

// const args = process.argv.splice(0, 2);
const NeuralNetwork = require("../build/src").NNetwork;
const process = require("process");
const fs = require("fs");

let nn = new NeuralNetwork(1, 2, 1);

// const weights = Array.from(nn.layers[1].neurons.map(v => Array.from(v.weights)));
// const biases = Array.from(nn.layers[1].neurons.map(v => v.bias));

if(!fs.existsSync(`${process.cwd()}/temp`))
  fs.mkdirSync(`${process.cwd()}/temp`);
nn.saveTo(`${process.cwd()}/temp/nn.n`);

nn = NeuralNetwork.loadFrom(`${process.cwd()}/temp/nn.n`);

// const weightsLoaded = Array.from(nn.layers[1].neurons.map(v => Array.from(v.weights)));
// const biasesLoaded = Array.from(nn.layers[1].neurons.map(v => v.bias));

// console.log(`Weights\nSaved: ${weights}, Loaded: ${weightsLoaded}`);
// console.log(`Biases\nSaved: ${biases}, Loaded: ${biasesLoaded}`);

console.log("Save-load test finished");
