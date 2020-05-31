const fs = require("fs");
const randomRange = 50;

class Neuron {
  value: number = 0;
  sum: number = 0;
  step: number = 1;
  weights: number[] = [];
  bias: number;
  prevLayer: Layer;

  //default act func is sigmoid
  activation = (x: number) => 1 / (1 + Math.exp(-x));
  actDerivative = (x: number) => Math.exp(-x) / ((1 + Math.exp(-x)) * (1 + Math.exp(-x)));

  //default cost func is difference;
  cost = (x: number, y: number) => x - y;

  constructor(prev?: Layer) {
    if (this.prevLayer) {
      this.prevLayer = prev;

      this.weights.length = prev.size;
      for (let i = 0; i < prev.size; ++i)
        this.weights[i] = (2 * Math.random() - 1) * randomRange;
    }

    this.bias = (2 * Math.random() - 1) * randomRange;
  }
  static copy(copyFrom: Neuron): Neuron {
    const c = new Neuron();

    c.step = copyFrom.step;
    c.weights = Array.from(copyFrom.weights);
    c.bias = copyFrom.bias;
    c.activation = copyFrom.activation;
    c.actDerivative = copyFrom.actDerivative;
    c.cost = copyFrom.cost;

    return c;
  }

  setValues() {
    if (!this.prevLayer) throw "prevLayer is undefined";

    this.sum = this.bias;
    for (let i = 0; i < this.prevLayer.size; ++i)
      this.sum += this.prevLayer.neurons[i].value * this.weights[i];
    this.value =  this.activation(this.sum);
  }

  change(diff: number): number[] {
    const biasDiff =
      this.cost(this.value + diff, this.value) * this.actDerivative(this.sum);
    this.bias += this.step * biasDiff;
    const prevDiff = [];

    for (let i = 0; i < this.prevLayer.size; ++i) {
      prevDiff.push(this.weights[i] * biasDiff);
      this.weights[i] += this.step * this.prevLayer.neurons[i].value * biasDiff;
    }

    return prevDiff;
  }
}

class Layer {
  neurons: Neuron[] = [];
  step: number = 1;
  next: Layer;

  constructor(size: number, prev?: Layer) {
    this.size = size;

    this.neurons.fill(new Neuron(prev));
  }

  set values(vals: number[]) {
    if (vals.length !== this.neurons.length) throw "set values invalid length";

    for (let i = 0; i < this.size; ++i)
      this.neurons[i].value = vals[i];
  }
  get values() {
    return this.neurons.map(v => v.value);
  }

  set size(val: number) {
    if (val < 0) throw "negative size";
    this.neurons.length = val;
  }
  get size() {
    return this.neurons.length;
  }

  set prev(val: Layer) {
    //sets new ref to prev layer and resets weights
    for (const n of this.neurons) n.prevLayer = val;
  }
  get prev() {
    return this.neurons[0].prevLayer;
  }

  //activation func for neurons
  set activation(f: (x: number) => number) {
    //activation func for neurons
    for (const n of this.neurons) n.activation = f;
  }
  set actDerivative(f: (x: number) => number) {
    //activation func derivative so it can be faster evaluated
    for (const n of this.neurons) n.actDerivative = f;
  }
  set cost(f: (x: number, y: number) => number) {
    //some cost function that will vanish at x = y
    for (const n of this.neurons) n.cost = f;
  }

  computeForward() {
    //if there is no prev just move forward
    if (this.prev) {
      for (let i = 0; i < this.size; ++i) {
        this.neurons[i].setValues();
      }
      //move forward only if next exists
    } else if (this.next) this.next.computeForward();
  }

  computeBackward(diff: number[]) {
    //if no prev then is's input layer
    //and nothing should be done
    if (!this.prev) return;

    let totalPrevDiff = new Array(this.prev.size).fill(0);

    for (let i = 0; i < this.size; ++i) {
      const prevDiff = this.neurons[i].change(diff[i]);

      totalPrevDiff = totalPrevDiff.map((v, i) => v + prevDiff[i]);
    }

    this.prev.computeBackward(totalPrevDiff);
  }

  removeNeuron(index: number) {
    this.neurons.splice(index, 1);

    if (this.next) for (const n of this.next.neurons) n.weights.splice(index, 1);
  }
  addNeuron() {
    this.neurons.push(new Neuron(this.prev));

    if (this.next) for (const n of this.next.neurons)
      n.weights.push((2 * Math.random() - 1) * randomRange);
  }

  static copy(copyFrom: Layer): Layer {
    let c: Layer;
    c.size = copyFrom.size;

    for (let i = 0; i < c.size; ++i) {
      c.neurons[i] = Neuron.copy(copyFrom.neurons[i]);
    }

    //no copying prev/next layer
    return c;
  }
}

class NNetwork {
  layers: Layer[] = [];

  constructor(...layout: number[]) {
    //layout is set of numbers of neurons per specific layer
    this.layers.length = layout.length;

    this.layers[0] = new Layer(layout[0]);
    for (let i = 1; i < this.layers.length; ++i) {
      this.layers[i] = new Layer(layout[i], this.layers[i - 1]);
      this.layers[i - 1].next = this.layers[i];
    }
  }

  set activation(f: (x: number) => number) {
    for (const l of this.layers) l.activation = f;
  }
  set actDerivative(f: (x: number) => number) {
    for (const l of this.layers) l.actDerivative = f;
  }
  set cost(f: (x: number, y: number) => number) {
    for (const l of this.layers) l.cost = f;
  }
  set step(val: number) {
    for (const l of this.layers) l.step = val;
  }

  get layout() {
    return this.layers.map(v => v.size);
  }

  static copy(copyFrom: NNetwork): NNetwork {
    const c: NNetwork = new NNetwork(...copyFrom.layout);

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

  feedforward(...input: number[]): number[] {
    this.layers[0].values = input;
    this.layers[0].computeForward();

    //copying so that references don't leak
    return this.layers[this.layers.length - 1].values;
  }
  backprop(...output: number[]) {
    //for backpropagation technique
    if (output.length !== this.layers[this.layers.length - 1].size)
      throw "backprop: invalid input";

    const diff = output.map(
      (out, i) => out - this.layers[this.layers.length - 1].values[i]
    );
    this.layers[this.layers.length - 1].computeBackward(diff);
  }

  //train on input-output pairs
  train(...examples: [number[], number[]][]) {
    for(const example of examples) {
      this.feedforward(...example[0]);
      this.backprop(...example[1]);
    }
  }
}

module.exports = { NNetwork, randomRange };
