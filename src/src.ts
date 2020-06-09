import * as fs from "fs";
const randomRange = 50;

class Neuron {
  value: number = 0;
  sum: number = 0;
  step: number = 1;
  weights: number[] = [];
  bias: number;
  private _prevLayer?: Layer;

  //default act func is sigmoid
  activation = (x: number) => 1 / (1 + Math.exp(-x));
  actDerivative = (x: number) => (this.activation(x + 0.0001) - this.activation(x - 0.0001)) / 0.0002;

  //default cost func is difference;
  cost = (x: number, y: number) => x - y;

  constructor(prev?: Layer) {
    if (prev)
      this.prevLayer = prev;

    this.bias = (2 * Math.random() - 1) * randomRange;
  }
  static copy(copyFrom: Neuron): Neuron {
    const buffer = new Neuron();

    buffer.step = copyFrom.step;
    buffer.weights = Array.from(copyFrom.weights);
    buffer.bias = copyFrom.bias;
    buffer.activation = copyFrom.activation;
    buffer.actDerivative = copyFrom.actDerivative;
    buffer.cost = copyFrom.cost;

    return buffer;
  }

  set prevLayer(prev: Layer) {
    this._prevLayer = prev;

    this.weights.length = prev.size;
    for (let i = 0; i < prev.size; ++i)
      if(this.weights[i] === undefined)
        this.weights[i] = (2 * Math.random() - 1) * randomRange;
  }
  get prevLayer() {
    return this._prevLayer!;
  }

  setValues() {
    if (!this.prevLayer) throw new Error("prevLayer is undefined");

    this.sum = this.bias;
    for (let i = 0; i < this.prevLayer.size; ++i)
      this.sum += this.prevLayer.neurons[i].value * this.weights[i];
    this.value =  this.activation(this.sum);
  }

  //changes neuron's parameters according to some expected difference
  change(diff: number) {
    //calculate diff as gradient
    const biasDiff =
      this.cost(this.value + diff, this.value) * this.actDerivative(this.sum);

    //if bias happens to be NaN do nothing (as if change is 0)
    if (isNaN(biasDiff)) return;

    this.bias += this.step * biasDiff;
    for (let i = 0; i < this.prevLayer.size; ++i)
      this.weights[i] += this.step * this.prevLayer.neurons[i].value * biasDiff;
  }

  //calculates changes to prevLayer's neurons based on its changes
  getPrevDiff(diff: number): number[] {
    let biasDiff =
      this.cost(this.value + diff, this.value) * this.actDerivative(this.sum);

    if (isNaN(biasDiff)) biasDiff = 0;
    const prevDiff = [];

    for (let i = 0; i < this.prevLayer.size; ++i)
      prevDiff.push(this.weights[i] * biasDiff);

    return prevDiff;
  }
}

class Layer {
  neurons: Neuron[] = [];
  next?: Layer;

  constructor(size: number, prev?: Layer) {
    if (size === 0) throw new Error("Zero size layer is useless");
    this.size = size;

    for (let i = 0; i < this.size; ++i)
      this.neurons[i] = new Neuron(prev);
  }
  static copy(copyFrom: Layer): Layer {
    const copy: Layer = new Layer(copyFrom.size);

    for (let i = 0; i < copy.size; ++i) {
      copy.neurons[i] = Neuron.copy(copyFrom.neurons[i]);
    }

    //no copying prev/next layer
    return copy;
  }

  set values(vals: number[]) {
    if (vals.length !== this.neurons.length)
      throw new Error("Setting values of invalid length");

    for (let i = 0; i < this.size; ++i)
      this.neurons[i].value = vals[i];
  }
  get values() {
    return this.neurons.map(v => v.value);
  }

  set size(val: number) {
    if (val < 1) throw new Error("Setting invalid size (size < 1)");
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
  set activation(func: (x: number) => number) {
    //activation func for neurons
    for (const n of this.neurons) n.activation = func;
  }
  set actDerivative(func: (x: number) => number) {
    //activation func derivative so it can be faster evaluated
    for (const n of this.neurons) n.actDerivative = func;
  }
  set cost(func: (x: number, y: number) => number) {
    //some cost function that will vanish at x = y
    for (const n of this.neurons) n.cost = func;
  }
  set step(val: number) {
    for (const n of this.neurons) n.step = val;
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

    //Change of values for prev layer
    const totalPrevDiff = new Array(this.prev.size).fill(0);

    //for each neuron computes its change and combined change for prev layer
    for (let i = 0; i < this.size; ++i) {
      this.neurons[i].change(diff[i]);
      this.neurons[i].getPrevDiff(diff[i]).forEach((v, i) => totalPrevDiff[i] += v);
    }

    this.prev.computeBackward(totalPrevDiff);
  }

  removeNeuron(index: number) {
    if (!this.neurons.length) throw new Error("There is 1 neuron, better remove whole layer");
    this.neurons.splice(index, 1);

    if (this.next) for (const n of this.next.neurons) n.weights.splice(index, 1);
  }
  addNeuron() {
    this.neurons.push(new Neuron(this.prev));

    if (this.next) for (const n of this.next.neurons)
      n.weights.push((2 * Math.random() - 1) * randomRange);
  }
}

class NNetwork {
  layers: Layer[] = [];

  constructor(...layout: number[]) {
    //layout is set of sizes per specific layer
    if(!layout) throw new Error("Empty layout");
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
    const copy: NNetwork = new NNetwork(...copyFrom.layout);

    for (let i = 0; i < copy.layers.length; ++i) {
      copy.layers[i] = Layer.copy(copyFrom.layers[i]);
      copy.layers[i - 1].next = copy.layers[i];
      copy.layers[i].prev = copy.layers[i - 1];
    }

    return copy;
  }
  static loadFrom(file: string): NNetwork {
    //create NN based on file
    const loaded: NNetwork = new NNetwork(1);
    const neuronsData = fs.readFileSync(file, "utf8").split("\n");
    //processing first iteration manually
    //to recover first layer's size

    for (let neuronDataI = 0; neuronDataI < neuronsData.length; ++neuronDataI) {
      if (neuronsData[neuronDataI].length) {
        const lastLayer = loaded.layers[loaded.layers.length - 1];
        lastLayer.addNeuron();
        const neuron = lastLayer.neurons[lastLayer.size - 1];

        const separatedValues = neuronsData[neuronDataI].split(" ");
        neuron.bias = parseFloat(separatedValues.shift()!);

        for (let i = 0; i < separatedValues.length; ++i)
          neuron.weights[i] = parseFloat(separatedValues[i]);

      } else {
        loaded.addLayer(loaded.layers.length, 1);

        ++neuronDataI;
        const lastLayer = loaded.layers[loaded.layers.length - 1];
        const neuron = lastLayer.neurons[lastLayer.size - 1];

        const separatedValues = neuronsData[neuronDataI].split(" ");
        neuron.bias = parseFloat(separatedValues.shift()!);

        if (neuronDataI === 1) loaded.layers[0].size = separatedValues.length;

        for (let i = 0; i < separatedValues.length; ++i)
          neuron.weights[i] = parseFloat(separatedValues[i]);
      }
    }
    return loaded;
  }
  saveTo(file: string) {
    //load NN to a file
    //\n\n delimits new layer
    let data = "";

    //input layer can be recovered from next to it layer
    const withoutInputLayer = Array.from(this.layers);
    withoutInputLayer.shift();
    withoutInputLayer.shift()?.neurons
      .forEach(v => data += "\n" + v.bias.toString() + " " + v.weights.toString().replace(/,/gi, " "));

    for (const l of withoutInputLayer) {
      data += "\n";
      // eslint-disable-next-line no-loop-func
      l.neurons.forEach(v => data += "\n" + v.bias.toString() + " " + v.weights.toString().replace(/,/gi, " "));
    }

    fs.writeFileSync(file, data);
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

  removeLayer(index: number) {
    this.layers.splice(index, 1);

    if (this.layers[index])
      this.layers[index].prev = this.layers[index - 1];
    this.layers[index - 1].next = this.layers[index];
  }
  addLayer(index: number, size: number) {
    this.layers.splice(index, 0, new Layer(size));

    if (this.layers[index - 1])
      this.layers[index - 1].next = this.layers[index];
    this.layers[index].prev = this.layers[index - 1];
    this.layers[index].next = this.layers[index + 1];
    if (this.layers[index + 1])
      this.layers[index + 1].prev = this.layers[index];
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
