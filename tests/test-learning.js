"use strict";

// const args = process.argv.splice(0, 2);
const NeuralNetwork = require("../build/src");

//try to learn linear func
const a = Math.random() * 10;
const b = Math.random() * 10;
const testFunc = x => a * x + b;

const nn = new NeuralNetwork([1, 1]);

nn.activation = x => x;
nn.actDerivative = () => 1;
nn.cost = (x, y) => x - y;

const startingW = [];
const startingB = Array.from(nn.layers[1].biases);

for (let i = 0; i < nn.layers[1].weights.length; ++i) {
  startingW[i] = [];
  for (let j = 0; j < nn.layers[1].weights[i].length; ++j) {
    startingW[i][j] = nn.layers[1].weights[i][j];
  }
}

//learning test
let input;

for(let i = 0; i < 100; ++i) {
  input = Math.random() * 1;
  for (let i = 0; i < 100; ++i) {
    nn.feedforward([input]);
    nn.backprop([testFunc(input)]);
  }
}

if (Math.abs(nn.feedforward([input]) - testFunc(input)) > 0.0001) {
  console.log("No overfitting one example\n");
  console.log(`testFunc: ${a} * x + ${b}`);
  console.log(`In: ${input}`);
  console.log(`Out: ${nn.feedforward([input])[0]} !== ${testFunc(input)}\n`);

  console.log("NN params:");
  console.log(nn.layers[1].weights);
  console.log(nn.layers[1].biases);
  console.log("\nNN starting params:");
  console.log(startingW);
  console.log(startingB);
  return;
}

//after learning weight and bias should match parameters a and b
//and outputs will match

input = Math.random() * 11;
if (nn.layers[1].weights[0][0] === a && nn.layers[1].biases[0] === b &&
    Math.abs(nn.feedforward([input]) - testFunc(input)) < 0.0001) {
  console.log("Passed");
} else {
  console.log(`In: ${input}`);
  console.log(`Out: ${nn.feedforward([input])[0]} !== ${testFunc(input)}\n`);
  console.log("Failed test");
  console.log(`Learned: a = ${nn.layers[1].weights[0][0]}, b = ${nn.layers[1].biases[0]}`);
  console.log(`Actual: a = ${a}, b = ${b}`);
}