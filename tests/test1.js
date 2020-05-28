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

const startingW = Array.from(nn.layers[1].weights);
const startingB = Array.from(nn.layers[1].biases);

//learning
let input;

//should overfit on one example
input = Math.random() * 10;

nn.feedforward([input]);
nn.backprop([testFunc(input)]);

if (nn.feedforward([input])[0] !== testFunc(input)) {
  console.log("No overfitting one example");
  console.log(`testFunc: ${a} * x + ${b}`);
  console.log(`In: ${input}`);
  console.log(`Out: ${nn.feedforward([input])[0]} !== ${testFunc(input)}`);

  console.log("NN params:");
  console.log(nn.layers[1].weights);
  console.log(nn.layers[1].biases);
  console.log("NN starting params:");
  console.log(startingW);
  console.log(startingB);
  return;
}

//after learning weight and bias should match parameters
//and outputs of testFunc and nn should alse match

input = 11;
if (nn.layers[1].weights[0][0] === a && nn.layers[1].biases[0] === b &&
    nn.feedforward([input]) === testFunc(input)) {
  console.log("Passed");
} else {
  console.log("Failed test");
  console.log(`Learned: a = ${nn.layers[1].weights[0][0]}, b = ${nn.layers[1].biases[0]}`);
  console.log(`Actual: a = ${a}, b = ${b}`);
}