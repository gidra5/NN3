class Layer {
    weights: number[][];
    biases:  number[];
    values:  number[];
    prev:    Layer;
    next:    Layer;

    activation:   (x: number) => number;
    removeNeuron: (index: number) => void;
    addNeuron:    () => void;

    Layer(size: number);
}

class NNetwork {
    #inputLayer:  Layer;
    #outputLayer: Layer;

    forwardprop:  (input: number[]) => number[]; //basic function for doing processing
    backprop:     (output: number[]) => number[];//for bacpropagation technique

    //dynamic layout adjustment
    removeLayer:  (index: number) => void;
    addLayer:     (size?: number) => void;
    removeNeuron: (layerIndex: number, neuronIndex: number) => void;
    addNeuron:    (layerIndex: number) => void;

    NNetwork(layout: number[]);
    NNetwork(copyFrom: NNetwork); //copy constructor
    NNetwork(file: string); //create NN based on file

}