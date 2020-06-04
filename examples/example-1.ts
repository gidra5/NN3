"use strict";

const NeuralNetwork = require("../src/src").NNetwork;
const fs = require("fs");

//Solving classic problem of learning XOR relation

const trainingData = [[[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]]];

const nn = new NeuralNetwork(2, 2, 1);

nn.train(...trainingData);

console.log(`1 XOR 0 = ${nn.feedforward(1, 0)}`);
console.log(`0.5 XOR 0.7 = ${nn.feedforward(0.5, 0.7)}`);

if(!fs.existsSync(`${process.cwd()}/temp`))
  fs.mkdirSync(`${process.cwd()}/temp`);
nn.saveTo(`${process.cwd()}/temp/example-1.nn`);