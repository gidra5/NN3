const fs = require("fs");
const randomRange = 50;

class Layer {
  weights: number[][]; //weights to the prev layer
  biases: number[];
  values: number[]; //values of neurons of this layer
  next: Layer;
  _prev: Layer;

  constructor(size: number, prev?: Layer) {
    this.size = size;

    for (let i = 0; i < this.size; ++i)
      this.values[i] = (2 * Math.random() - 1) * randomRange;
    for (let i = 0; i < this.size; ++i)
      this.biases[i] = (2 * Math.random() - 1) * randomRange;
    for (let i = 0; i < this.size; ++i) this.weights[i] = [];

    if (prev) this.prev = prev;
  }

  get size() {
    //amount of neurons
    return this.values.length;
  }

  set size(val: number) {
    this.values.length = val;
    this.biases.length = val;
    this.weights.length = val;
  }

  set prev(val: Layer) {
    //sets new ref to prev layer and resets weights
    this._prev = val;

    for (let i = 0; i < this.size; ++i) {
      this.weights[i].length = this._prev.size;
      this.weights[i] = this.weights[i].map(
        () => (2 * Math.random() - 1) * randomRange
      );
    }
  }

  //activation func for neurons
  activation: (x: number) => number;
  //derivative of activation func for backprop
  actDerivative = (x: number) =>
    (this.activation(x + 0.0001) - this.activation(x - 0.0001)) / 0.0002;
  //some cost function that will vanish at x = y
  cost: (x: number, y: number) => number;

  computeForward() {
    let sum: number;

    for (let i = 0; i < this.size; ++i) {
      sum = 0;
      for (let j = 0; j < this.prev.size; ++j)
        sum += this.weights[i][j] * this.prev.values[j];
      this.values[i] = this.activation(sum);
    }

    if (this.next) this.next.computeForward();
  }
  computeBackward(diff: number[]) {
    const biasesDiff = [];
    const prevDiff = new Array(this._prev.size).fill(0);

    let sum: number;
    for (let i = 0; i < this.size; ++i) {
      sum = 0;
      for (let j = 0; j < this._prev.size; ++j)
        sum += this.weights[i][j] * this.prev.values[j];

      biasesDiff[i] =
        this.cost(this.values[i] + diff[i], this.values[i]) *
        this.actDerivative(sum);
      this.biases[i] += biasesDiff[i];

      for (let j = 0; j < this._prev.size; ++j) {
        this.weights[i][j] += this._prev.values[j] * biasesDiff[i];
        prevDiff[i] += this.weights[i][j] * biasesDiff[i];
      }
    }

    if (this._prev) this._prev.computeBackward(prevDiff);
  }

  removeNeuron(index: number) {
    this.weights.splice(index, 1);
    this.values.splice(index, 1);
    this.biases.splice(index, 1);

    if (this.next) for (const w of this.next.weights) w.splice(index, 1);
  }
  addNeuron() {
    this.biases.push((2 * Math.random() - 1) * randomRange);
    this.weights.push(
      new Array(this._prev.size).map(
        () => (2 * Math.random() - 1) * randomRange
      )
    );
    this.values.push(0);

    if (this.next)
      for (const w of this.next.weights)
        w.push((2 * Math.random() - 1) * randomRange);
  }

  static copy(copyFrom: Layer): Layer {
    const c: Layer;
    c.size = copyFrom.size;

    for (let i = 0; i < c.size; ++i) {
      c.weights[i].length = copyFrom.weights[0].length;
    }

    //todo

    return c;
  }
}

class NNetwork {
  layers: Layer[];

  constructor(layout: number[]) {
    //layout is set of numbers of neurons per specific layer
    this.layers.length = layout.length;
    for (let i = 0; i < this.layers.length; ++i) {
      this.layers[i] = new Layer(layout[i]);
      this.layers[i - 1].next = this.layers[i];
      this.layers[i].prev = this.layers[i - 1];
    }

    //default act func is sigmoid
    this.activation = (x) => 1 / (1 - Math.exp(x));

    //default cost func is difference;
    this.cost = (x, y) => x - y;
  }

  set activation(f: (x: number) => number) {
    //activation func for neurons
    for (const l of this.layers) l.activation = f;
  }
  set cost(f: (x: number, y: number) => number) {
    //some cost function that will vanish at x = y
    for (const l of this.layers) l.cost = f;
  }
  get layout() {
    const l = [];
    this.layers.forEach((layer) => l.push(layer.size));
    return l;
  }

  static copy(copyFrom: NNetwork): NNetwork {
    const c: NNetwork = new NNetwork(copyFrom.layout);

    for (let i = 0; i < c.layers.length; ++i)
      c.layers[i] = Layer.copy(copyFrom.layers[i]);

    return c;
  }
  static loadFrom(file: string): NNetwork {
    //create NN based on file
    let loaded: NNetwork;
    fs.readFile(file, (err, data) => {
      if (err) throw err;

      loaded = JSON.parse(data);
    });

    return loaded;
  }
  saveTo(file: string) {
    //load NN to a file lazy way
    const data = JSON.stringify(this);

    fs.writeFile(file, data, (err) => {
      if (err) throw err;
    });
  }

  forwardprop(input: number[]): number[] {
    if (input.length !== this.layers[0].size)
      throw "forwardprop: invalid input";

    this.layers[0].values = input;
    this.layers[0].computeForward();

    //copying so that references don't leak
    return Array.from(this.layers[this.layers.length - 1].values);
  }
  backprop(output: number[]) {
    //for backpropagation technique
    if (output.length !== this.layers[this.layers.length - 1].size)
      throw "backprop: invalid input";

    const diff = output.map(
      (out, i) => out - this.layers[this.layers.length - 1].values[i]
    );
    this.layers[this.layers.length - 1].computeBackward(diff);
  }
}

module.exports = NNetwork;
