# Documentation

## Neuron

Basic computational unit of neural networks  

### Fields

Field | Desription 
----- | ---------- 
`value: number` | holds calculated for this neuron value
`bias: number` | holds neuron's bias. Initialized with random value
`weights: number` | holds weights for incoming values. It's length is equal to the previous layer's size (0 if no such). Initialized with random values
`step: number` | learning rate of neuron
`activation: (number) => number` | activation function for neuron. Default is sigmoid function.
`actDerivative: (number) => number` | activation function's derivative for neuron. Needed to get precise derivative. Default is numeric derivative with step 0.0001.

### Methods

Method | Desription 
------ | ---------- 
`constructor(prev?: Layer): Neuron` | constructor of Neuron class. Optionally takes `prev` to set `weights` array's length
`static copy(copyFrom: Neuron): Neuron` | makes copy of a neuron from `copyFrom` neuron. Copies all fields except for `value`.
`setValues()` | calculates values of this neuron.
`change(diff: number)` | changes weights and bias to change `value` by approximately `diff` amount
`getPrevDiff(diff: number): number[]` | calculates how input should change to change `value` by approximately `diff` amount. Size of returned array is equal to `weights` length

## Layer

Organizes work with arrays of neurons in neural networks

### Fields

Field | Desription 
----- | ---------- 
`neurons: Neuron[]` |  neurons that this layer contains and operates on
`next: Layer` | next to this layer in NN

### Methods

Method | Desription 
------ | ---------- 
`constructor(size: number, prev?: Layer): Layer` | constructor of Layer. Takes `size` to specify amount of neurons on layer and optionaly `prev` to be set for the neurons
`static copy(copyFrom: Layer): Layer` | makes copy of `copyFrom` layer. Copies all fields of Layer class.
`set values(vals: number[])` | setter for `value` field of each neuron according to given `vals`
`get values(): number[]` |  getter for `value` field of each neuron in layer organized in array
`set size(val: number)` | setter for size of layer. Just sets length of `neurons` array and throws error if `size < 1`
`get size(): number` | getter for size of this layer
`set prev(val: Layer)` | setter for previous to this layer in NN
`get prev(): Layer` | setter for previous to this layer in NN
`set activation(func: (x: number) => number)` | setter for activation function for all neurons of layer.
`set actDerivative(func: (x: number) => number)` | setter for activation function derivative for all neurons of layer.
`set step(val: number)` | setter for `step` field for all neurons of layer.
`computeForward()` | computes values of this layer and next to it layer.
`computeBackward(diff: number[])` | computes changes to neurons given deired change in `values`
`removeNeuron(index: number)` | removes neuron from this layer and makes needen changes to next and previous layers
`addNeuron()` | adds new neuron to the layer and makes needen changes to next and previous layers

## NNetwork

Basic feedforward neural network implementation

### Fields

Field | Desription 
----- | ---------- 
`layers: Layer[]` | layers of this NN

### Methods

Method | Desription 
------ | ---------- 
`constructor(...layout: number[])` | constructs neural network from its layout 
`static copy(copyFrom: NNetwork): NNetwork` | makes copy neural network from `copyFrom`. Copies all fields of `copyFrom` network
`static loadFrom(file: string): NNetwork` | loads nn from `file` file
`saveTo(file: string)` | saves this neural network to `file` file
`set activation(f: (x: number) => number)` | setter for activation function for all neurons of NN.
`set actDerivative(f: (x: number) => number)` | setter for activation function derivative for all neurons of NN.
`set step(val: number)` | setter for `step` field for all neurons of NN.
`get layout()` | getter for layout of this NN
`feedforward(...input: number[]): number[]` | feedsforward `input` to calculate values of neurons and output of NN
`backprop(...output: number[])` | backpropagates to change NN so that it's closer to desired `output`
`removeLayer(index: number)` | removes layer at given `index` and makes needed changes to adjacent layers
`addLayer(index: number, size: number)` | inserts new layer of given`size` at given `index` and makes needed changes to adjacent layers
`train(...examples: [number[], number[]][])` | trains NN on passed examples (pairs of input-output arrays)
