"use strict";

// const args = process.argv.splice(0, 2);
const NeuralNetwork = require("../build/src").NNetwork;

//try to learn linear func
const a = 5;
const b = 10;
const testFunc = x => a * x + b;

const nn = new NeuralNetwork(1, 1);

nn.activation = x => x;
nn.actDerivative = () => 1;

const startingW = Array.from(nn.layers[1].neurons.map(v => v.weights));
const startingB = Array.from(nn.layers[1].neurons.map(v => v.bias));

//learning test
let input = Math.random() * 1;
nn.feedforward([input]);
nn.backprop([testFunc(input)]);

console.log(`testFunc: ${a} * x + ${b}`);
console.log(`In: ${input}`);
console.log(`Out: ${nn.feedforward([input])[0]} !== ${testFunc(input)} (exact)`);
console.log(`Error: ${nn.feedforward([input])[0] - testFunc(input)}`);
console.log(`Error a: ${nn.layers[1].neurons[0].weights[0] - a}`);
console.log(`Error b: ${nn.layers[1].neurons[0].bias - b}`);

console.log("\nNN params:");
console.log(nn.layers[1].neurons.map(v => v.weights));
console.log(nn.layers[1].neurons.map(v => v.bias));
console.log("\nNN starting params:");
console.log(startingW);
console.log(startingB);

console.log("\nLearning test finished successfully");