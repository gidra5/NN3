const fs = require("fs");
const randomRange = 50;

class Layer {
  weights: number[][] = []; //weights to the prev layer
  biases: number[] = [];
  values: number[] = []; //values of neurons of this layer
  step: number = 1;
  _prev: Layer;
  next: Layer;

  constructor(size: number) {
    this.size = size;

    for (let i = 0; i < this.size; ++i)
      this.values[i] = 0;
    for (let i = 0; i < this.size; ++i)
      this.biases[i] = (2 * Math.random() - 1) * randomRange;
    for (let i = 0; i < this.size; ++i) this.weights[i] = [];
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
      for (let j = 0; j < this.weights[i].length; ++j)
        this.weights[i][j] = (2 * Math.random() - 1) * randomRange;
    }
  }

  //activation func for neurons
  activation: (x: number) => number;
  //derivative of activation func for backprop
  //default is numerical derivative, that should be replaced by exact func
  actDerivative = (x: number) =>
    (this.activation(x + 0.0001) - this.activation(x - 0.0001)) / 0.0002;
  //some cost function that will vanish at x = y
  cost: (x: number, y: number) => number;

  computeForward() {
    //if there is no prev just move forward
    if (this._prev) {
      let sum: number;

      for (let i = 0; i < this.size; ++i) {
        sum = this.biases[i];
        for (let j = 0; j < this._prev.size; ++j)
          sum += this.weights[i][j] * this._prev.values[j];
        this.values[i] = this.activation(sum);
      }
      //move forward only if next exists
    } else if (this.next) this.next.computeForward();
  }
  computeBackward(diff: number[]) {
    //if no prev then is's input layer
    //and nothing should be done
    if (!this._prev) return;

    const biasesDiff = [];
    const prevDiff = new Array(this._prev.size).fill(0);

    let sum: number;
    for (let i = 0; i < this.size; ++i) {
      sum = this.biases[i];
      for (let j = 0; j < this._prev.size; ++j)
        sum += this.weights[i][j] * this._prev.values[j];

      biasesDiff[i] =
        this.cost(this.values[i] + diff[i], this.values[i]) *
        this.actDerivative(sum);
      this.biases[i] += this.step * biasesDiff[i];

      for (let j = 0; j < this._prev.size; ++j) {
        this.weights[i][j] += this.step * this._prev.values[j] * biasesDiff[i];
        prevDiff[i] += this.step * this.weights[i][j] * biasesDiff[i];
      }
    }

    this._prev.computeBackward(prevDiff);
  }

  removeNeuron(index: number) {
    this.weights.splice(index, 1);
    this.values.splice(index, 1);
    this.biases.splice(index, 1);

    if (this.next) for (const w of this.next.weights) w.splice(index, 1);
  }
  addNeuron() {
    this.biases.push((2 * Math.random() - 1) * randomRange);

    if (this._prev)
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
    let c: Layer;
    c.size = copyFrom.size;

    //copying values of weights and biases
    for (let i = 0; i < c.size; ++i) {
      c.weights[i].length = copyFrom.weights[i].length;
      for (let j = 0; j < c.size; ++j)
        c.weights[i][j] = copyFrom.weights[i][j];

      c.biases[i] = copyFrom.biases[i];
    }

    //and copying functions
    c.activation = copyFrom.activation;
    c.cost = copyFrom.cost;

    //but don't copy prev/next layer
    return c;
  }
}

class NNetwork {
  layers: Layer[] = [];

  constructor(layout: number[]) {
    //layout is set of numbers of neurons per specific layer
    this.layers.length = layout.length;

    this.layers[0] = new Layer(layout[0]);
    for (let i = 1; i < this.layers.length; ++i) {
      this.layers[i] = new Layer(layout[i]);
      this.layers[i - 1].next = this.layers[i];
      this.layers[i].prev = this.layers[i - 1];
    }

    //default act func is sigmoid
    this.activation = x => 1 / (1 + Math.exp(-x));
    this.actDerivative = x => Math.exp(-x) / ((1 + Math.exp(-x)) * (1 + Math.exp(-x)));

    //default cost func is difference;
    this.cost = (x, y) => x - y;
  }

  set activation(f: (x: number) => number) {
    //activation func for neurons
    for (const l of this.layers) l.activation = f;
  }
  set actDerivative(f: (x: number) => number) {
    //activation func for neurons
    for (const l of this.layers) l.actDerivative = f;
  }
  set cost(f: (x: number, y: number) => number) {
    //some cost function that will vanish at x = y
    for (const l of this.layers) l.cost = f;
  }
  set step(val: number) {
    for (const l of this.layers) l.step = val;
  }

  get layout() {
    const l = [];
    this.layers.forEach(layer => l.push(layer.size));
    return l;
  }

  static copy(copyFrom: NNetwork): NNetwork {
    const c: NNetwork = new NNetwork(copyFrom.layout);

    for (let i = 0; i < c.layers.length; ++i) {
      c.layers[i] = Layer.copy(copyFrom.layers[i]);
      c.layers[i - 1].next = c.layers[i];
      c.layers[i].prev = c.layers[i - 1];
    }

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

    fs.writeFile(file, data, err => {
      if (err) throw err;
    });
  }

  feedforward(input: number[]): number[] {
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
