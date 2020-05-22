const fs = require('fs');
const randomRange = 50;

class Layer {
    weights: number[][]; //weights to the prev layer
    biases:  number[];
    values:  number[];  //values of neurons of this layer
    next:    Layer;
    _prev:   Layer;

    constructor(size: number) {
        this.values.length = size;
        this.biases.length = size;
        this.weights.length = size;
        for(let v of this.values)
            v = (2 * Math.random() - 1) * randomRange;
        for(let b of this.biases)
            b = (2 * Math.random() - 1) * randomRange;
        for(let w of this.weights)
            w = [];
    };

    get size() { //amount of neurons
        return this.values.length;
    }

    set prev(val: Layer) {
        this._prev = val;

        for (let w of this.weights)
            w = new Array(this._prev.size).map(v => (2 * Math.random() - 1) * randomRange);
    }

    activation: (x: number) => number; //activation func for neurons
    actDerivative = //derivative of activation func for backprop
        (x: number) => (this.activation(x + 0.0001) - this.activation(x - 0.0001)) / 0.0002;
    cost: (x: number, y: number) => number; //some cost function that will vanish at x = y

    computeForward() {
        let sum = Array.from(this.values).fill(0);
        for (let i = 0; i < this.size; ++i) {
            for (let j = 0; j < this.prev.size; ++j)
            sum[i] += this.weights[i][j] * this.prev.values[j];
        }
        this.values = sum.map(v => this.activation(v));

        if (this.next) this.next.computeForward();
    };
    computeBackward(diff: number[]) {
        const biasesDiff = [];
        const prevDiff = new Array(this.prev.size).fill(0);

        let sum = 0;
        for (let i = 0; i < this.size; ++i) {
            sum = 0;
            for (let j = 0; j < this.prev.size; ++j)
                sum += this.weights[i][j] * this.prev.values[j];
            biasesDiff[i] = this.cost(this.values[i] + diff[i], this.values[i]) * this.actDerivative(sum);
        }

        for (let i = 0; i < this.weights.length; ++i) {
            for (let j = 0; j < this.weights[i].length; ++j) {
                this.weights[i][j] += this._prev.values[j] * biasesDiff[i];
                prevDiff[i] += this.weights[i][j] * biasesDiff[i];
            }
            this.biases[i] += biasesDiff[i];
        }

        if(this._prev)
            this._prev.computeBackward(prevDiff);
    }

    removeNeuron(index: number) {
        this.weights.splice(index, 1);
        this.values.splice(index, 1);
        this.biases.splice(index, 1);

        if(this.next)
            for (const w of this.next.weights)
                w.splice(index, 1);
    };
    addNeuron() {
        this.weights.push(new Array(this._prev.size).map(v => (2 * Math.random() - 1) * randomRange));
        this.values.push(0);
        this.biases.push((2 * Math.random() - 1) * randomRange);

        if(this.next)
            for (const w of this.next.weights)
                w.push((2 * Math.random() - 1) * randomRange);
    };

    copy(copyFrom: Layer): Layer {
        this.values.length = copyFrom.size;
        this.biases.length = copyFrom.size;
        this.weights.length = copyFrom.size;

        for (let w of this.weights)
            w.length = copyFrom.weights[0].length;

        //todo

        return this;
    }
}

class NNetwork {
    layers: Layer[];

    constructor(layout: number[]) { //number of neurons per specific layer
        this.layers.length = layout.length;
        for (let i = 0; i < this.layers.length; ++i) {
            this.layers[i] = new Layer(layout[i]);
            this.layers[i - 1].next = this.layers[i];
            this.layers[i].prev = this.layers[i - 1];
        }
    };

    set activation(f: (x: number) => number) { //activation func for neurons
        for (const l of this.layers)
            l.activation = f;
    };
    set cost(f: (x: number, y: number) => number) { //some cost function that will vanish at x = y
        for (const l of this.layers)
            l.cost = f;
    }


    copy(copyFrom: NNetwork): NNetwork {
        this.layers.length = copyFrom.layers.length;
        for (let i = 0; i < this.layers.length; ++i) {
            this.layers[i] = new Layer(copyFrom.layers[i].size).copy(copyFrom.layers[i]);
            this.layers[i - 1].next = this.layers[i];
            this.layers[i].prev = this.layers[i - 1];
        }
        return this;
    };
    loadFrom(file: string) { //create NN based on file
        fs.readFile(file, (err, data) => {
            if (err) throw err;
            //todo
        })
    };
    saveTo(file: string) { //load NN to a file
        let data = "";

        //todo

        fs.writeFile(file, data, (err) => {
            if (err) throw err;
        });
    };

    forwardprop(input: number[]): number[] {
        //to make assertions easier return invalid value in case of bad input
        if (input.length !== this.layers[0].size) throw 'forwardprop: invalid input';

        this.layers[0].values = input;
        this.layers[0].computeForward();

        return Array.from(this.layers[this.layers.length - 1].values);
    };
    backprop(output: number[]) { //for backpropagation technique
        if (output.length !== this.layers[this.layers.length - 1].size) throw 'backprop: invalid input';

        const diff = output.map((out, i) => out - this.layers[this.layers.length - 1].values[i])
        this.layers[this.layers.length - 1].computeBackward(diff);
    };
}

module.exports = NNetwork;