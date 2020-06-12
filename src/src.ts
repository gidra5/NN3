import * as fs from "fs";
const randomRange = 50;

class Neuron {
  step: number = 1;
  value: number = 0;
  bias: number;

  //weights created through proxy
  //because its better to populate possible empty elements with random values
  weights: number[] = new Proxy<number[]>([], {
    set: (obj, prop, val): boolean => {
      obj[prop] = val;

      if (prop === "length") {
        for (let i = 0; i < obj.length; ++i)
          if (obj[i] === undefined)
            obj[i] = (2 * Math.random() - 1) * randomRange;
      }

      return true;
    },
  });

  private sum: number = 0;

  activation = (x: number) => 1 / (1 + Math.exp(-x));
  actDerivative = (x: number) =>
    (this.activation(x + 0.0001) - this.activation(x - 0.0001)) / 0.0002;

  constructor(prevLayer?: Layer) {
    this.bias = (2 * Math.random() - 1) * randomRange;

    if (prevLayer) this.weights.length = prevLayer.size;
  }
  static copy(copyFrom: Neuron): Neuron {
    const copy = new Neuron();

    copy.step = copyFrom.step;
    copy.weights = Array.from(copyFrom.weights);
    copy.bias = copyFrom.bias;
    copy.activation = copyFrom.activation;
    copy.actDerivative = copyFrom.actDerivative;

    return copy;
  }

  setValues(prevValues: number[]): void {
    if (prevValues.length !== this.weights.length)
      throw new Error("prevLayer's size isn't matching weights' length");

    this.sum = this.bias;
    for (let i = 0; i < prevValues.length; ++i)
      this.sum += prevValues[i] * this.weights[i];
    this.value = this.activation(this.sum);
  }

  change(change: number, prevValues: number[]): void {
    if (prevValues.length !== this.weights.length)
      throw new Error("prevLayer's size isn't matching weights' length");

    const biasChange = this.step * change * this.actDerivative(this.sum);

    //if biasChange happens to be NaN do nothing (as if change is 0)
    if (isNaN(biasChange)) return;

    this.bias += biasChange;

    for (let i = 0; i < this.weights.length; ++i)
      this.weights[i] += prevValues[i] * biasChange;
  }

  getPrevChange(change: number): number[] {
    let biasChange = change * this.actDerivative(this.sum);

    if (isNaN(biasChange)) biasChange = 0;

    return this.weights.map(v => v * biasChange);
  }
}

class Layer {
  neurons: Neuron[] = [];
  prev?: Layer;
  next?: Layer;

  constructor(size: number, prev?: Layer) {
    if (size < 1) throw new Error("Size should be positive integer");
    this.prev = prev;
    this.size = size;
  }
  static copy(copyFrom: Layer): Layer {
    const copy: Layer = new Layer(copyFrom.size);

    for (let i = 0; i < copy.size; ++i)
      copy.neurons[i] = Neuron.copy(copyFrom.neurons[i]);

    return copy;
  }

  set values(vals: number[]) {
    if (vals.length !== this.size)
      throw new Error("Setting values of invalid length");

    for (let i = 0; i < this.size; ++i) this.neurons[i].value = vals[i];
  }
  get values() {
    return this.neurons.map(v => v.value);
  }

  set size(val: number) {
    if (val < 1) throw new Error("Setting invalid size (size < 1)");
    this.neurons.length = val;
    for (let i = 0; i < this.size; ++i) this.neurons[i] = new Neuron(this.prev);
  }
  get size() {
    return this.neurons.length;
  }

  set activation(func: (x: number) => number) {
    this.neurons.forEach(n => n.activation = func);
  }
  set actDerivative(func: (x: number) => number) {
    this.neurons.forEach(n => n.actDerivative = func);
  }
  set step(val: number) {
    this.neurons.forEach(n => n.step = val);
  }

  computeForward(): void {
    this.neurons.forEach(v => v.setValues(this.prev!.values));
    this.next?.computeForward();
  }

  computeBackward(diff: number[]): void {
    if (!this.prev) return;

    const totalPrevChange = new Array(this.prev.size).fill(0);

    for (let i = 0; i < this.size; ++i) {
      this.neurons[i].getPrevChange(diff[i]).forEach((v, j) => (totalPrevChange[j] += v));
      this.neurons[i].change(diff[i], this.prev.values);
    }

    this.prev.computeBackward(totalPrevChange);
  }

  removeNeuron(index: number): void {
    if (this.size < 2)
      throw new Error("There are 1 or 0 neurons, better remove whole layer");

    this.neurons.splice(index, 1);

    this.next?.neurons.forEach(n => n.weights.splice(index, 1));
  }
  addNeuron(): void {
    this.neurons.push(new Neuron(this.prev));

    this.next?.neurons.forEach(n => n.weights.push((2 * Math.random() - 1) * randomRange));
  }
}

class NNetwork {
  layers: Layer[] = [];

  constructor(...layout: number[]) {
    if (!layout.length) throw new Error("Empty layout");
    this.layers.length = layout.length;

    this.layers[0] = new Layer(layout[0]);
    for (let i = 1; i < this.layers.length; ++i) {
      this.layers[i] = new Layer(layout[i], this.layers[i - 1]);
      this.layers[i - 1].next = this.layers[i];
    }
  }

  set activation(func: (x: number) => number) {
    this.layers.forEach(l => l.activation = func);
  }
  set actDerivative(func: (x: number) => number) {
    this.layers.forEach(l => l.actDerivative = func);
  }
  set step(val: number) {
    this.layers.forEach(l => l.step = val);
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
    const loaded: NNetwork = new NNetwork(1);
    const neuronsData = fs.readFileSync(file, "utf8").split("\n");

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
  saveTo(file: string): void {
    let data = "";

    //input layer can be recovered from next to it layer
    const withoutInputLayer = Array.from(this.layers);
    withoutInputLayer.shift();
    withoutInputLayer.shift()?.neurons.
      forEach(v => data += "\n" + v.bias.toString() + " " + v.weights.toString().replace(/,/gi, " "));

    for (const l of withoutInputLayer) {
      data += "\n";
      // eslint-disable-next-line no-loop-func
      l.neurons.forEach(v => data += "\n" + v.bias.toString() + " " + v.weights.toString().replace(/,/gi, " "));
    }

    fs.writeFileSync(file, data);
  }

  feedforward(...input: number[]): number[] {
    this.layers[0].values = input;
    this.layers[1].computeForward();

    return this.layers[this.layers.length - 1].values;
  }
  backprop(...output: number[]): void {
    if (output.length !== this.layers[this.layers.length - 1].size)
      throw new Error("Invalid input size");

    const changes = output.map(
      (out, i) => out - this.layers[this.layers.length - 1].values[i]
    );
    this.layers[this.layers.length - 1].computeBackward(changes);
  }

  removeLayer(index: number): void {
    this.layers.splice(index, 1);

    if (this.layers[index]) this.layers[index].prev = this.layers[index - 1];
    this.layers[index - 1].next = this.layers[index];
  }
  addLayer(index: number, size: number): void {
    this.layers.splice(index, 0, new Layer(size));

    if (this.layers[index - 1])
      this.layers[index - 1].next = this.layers[index];

    this.layers[index].prev = this.layers[index - 1];
    this.layers[index].next = this.layers[index + 1];

    if (this.layers[index + 1])
      this.layers[index + 1].prev = this.layers[index];
  }

  train(...examples: [number[], number[]][]): void {
    for (const example of examples) {
      this.feedforward(...example[0]);
      this.backprop(...example[1]);
    }
  }
}

module.exports = { NNetwork, randomRange };
