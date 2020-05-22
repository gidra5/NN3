const fs = require('fs');
const randomRange = 50;
class Layer {
    constructor(size) {
        this.actDerivative = (x) => (this.activation(x + 0.0001) - this.activation(x - 0.0001)) / 0.0002;
        this.values.length = size;
        this.biases.length = size;
        this.weights.length = size;
        for (let v of this.values)
            v = (2 * Math.random() - 1) * randomRange;
        for (let b of this.biases)
            b = (2 * Math.random() - 1) * randomRange;
        for (let w of this.weights)
            w = [];
    }
    ;
    get size() {
        return this.values.length;
    }
    set prev(val) {
        this._prev = val;
        for (let w of this.weights)
            w = new Array(this._prev.size).map(v => (2 * Math.random() - 1) * randomRange);
    }
    computeForward() {
        let sum = Array.from(this.values).fill(0);
        for (let i = 0; i < this.size; ++i) {
            for (let j = 0; j < this.prev.size; ++j)
                sum[i] += this.weights[i][j] * this.prev.values[j];
        }
        this.values = sum.map(v => this.activation(v));
        if (this.next)
            this.next.computeForward();
    }
    ;
    computeBackward(diff) {
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
        if (this._prev)
            this._prev.computeBackward(prevDiff);
    }
    removeNeuron(index) {
        this.weights.splice(index, 1);
        this.values.splice(index, 1);
        this.biases.splice(index, 1);
        if (this.next)
            for (const w of this.next.weights)
                w.splice(index, 1);
    }
    ;
    addNeuron() {
        this.weights.push(new Array(this._prev.size).map(v => (2 * Math.random() - 1) * randomRange));
        this.values.push(0);
        this.biases.push((2 * Math.random() - 1) * randomRange);
        if (this.next)
            for (const w of this.next.weights)
                w.push((2 * Math.random() - 1) * randomRange);
    }
    ;
    copy(copyFrom) {
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
    constructor(layout) {
        this.layers.length = layout.length;
        for (let i = 0; i < this.layers.length; ++i) {
            this.layers[i] = new Layer(layout[i]);
            this.layers[i - 1].next = this.layers[i];
            this.layers[i].prev = this.layers[i - 1];
        }
    }
    ;
    copy(copyFrom) {
        this.layers.length = copyFrom.layers.length;
        for (let i = 0; i < this.layers.length; ++i) {
            this.layers[i] = new Layer(copyFrom.layers[i].size).copy(copyFrom.layers[i]);
            this.layers[i - 1].next = this.layers[i];
            this.layers[i].prev = this.layers[i - 1];
        }
        return this;
    }
    ;
    loadFrom(file) {
        fs.readFile(file, (err, data) => {
        });
    }
    ;
    saveTo(file) {
        let data = "";
        //todo
        fs.writeFile(file, data, (err) => {
            if (err)
                throw err;
        });
    }
    ;
    forwardprop(input) {
        //to make assertions easier return invalid value in case of bad input
        if (input.length !== this.layers[0].size)
            return undefined;
        this.layers[0].values = input;
        this.layers[0].computeForward();
        return Array.from(this.layers[this.layers.length - 1].values);
    }
    ;
    backprop(output) {
        //to make assertions easier return invalid value in case of bad input
        if (output.length !== this.layers[this.layers.length - 1].size)
            return undefined;
        const diff = output.map((out, i) => out - this.layers[this.layers.length - 1].values[i]);
        this.layers[this.layers.length - 1].computeBackward(diff);
        return 1;
    }
    ;
}
module.exports = NNetwork;
