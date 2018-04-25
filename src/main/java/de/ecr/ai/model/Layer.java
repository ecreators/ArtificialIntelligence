package de.ecr.ai.model;

import de.ecr.ai.model.annotation.LearningData;
import de.ecr.ai.model.neuron.*;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static java.text.MessageFormat.format;
import static java.util.stream.Collectors.toList;

/**
 * Defines a cluster to neurons to align a related solution path for the neural network
 *
 * @author Bjoern Frohberg
 */
public final class Layer {

  private final String name;
    private final List<Neuron> neurons;
    private NeuronType type;
    private final NeuralNetwork network; // for later commits "back propagation"

    public Layer(String name, NeuralNetwork network) {
        this.name = name;
        this.network = network;
        this.neurons = new ArrayList<>();
    }

    /**
     * Returns an error value for a given parent neuron in connection (binding)
     */
    private float calculateHiddenError(Neuron parentNeuron) {
        return (float) neurons.stream()
                .mapToDouble(childNeuron -> childNeuron.calculateParentError(parentNeuron))
                .sum();
    }

    /**
     * Appends new neurons (no softmax on output)
     */
    public void createNeurons(int neuronsCount, NeuronType type, boolean softmax) {
        this.type = type;

        Function<Integer, Neuron> builder = detectNeuronBuilder(type, softmax);

        for (int i = 0; i < neuronsCount; i++) {
            Neuron neuron = builder.apply(i);
            if (neuron == null) {
                throw new RuntimeException("Builder invalid! It created a null-neuron!");
            }
            neurons.add(neuron);
        }

        // to append bindFullMesh
        int removedNeurons = 0;
        for (int i = 0; i < neurons.size(); i++) {
            if (neurons.get(i) == null) {
                neurons.remove(i--);
                removedNeurons++;
            }
        }
        if (removedNeurons > 0) {
            throw new RuntimeException(removedNeurons + " null-neurons built!");
        }

        if (neuronsCount > neurons.size()) {
            throw new RuntimeException("Issues during creating neurons! count not as expected");
        }
    }

    /**
     * Returns the layer neuron type used to create neurons on this layer
     */
    public NeuronType getType() {
        return type;
    }

    /**
     * Bind any neuron in this layer to each neuron in the parent layer neurons as a fully mesh.
     * Requires a layer of type {@link NeuronType#INPUT} or {@link NeuronType#HIDDEN}
     */
    public Layer bindFullMesh(Layer parentLayer) {
        if (parentLayer.getType() != NeuronType.OUTPUT) {
            if (neurons.isEmpty()) {
                throw new RuntimeException("Cannot bind on empty set of neurons. Neurons undefined!");
            }
            int i = 0;
            for (Neuron neuron : neurons) {
                if (neuron instanceof IPropagateBack) {
                    bindToLayerNeurons(parentLayer, (IPropagateBack) neuron);
                    i++;
                }
            }
            if (i < neurons.size()) {
                throw new RuntimeException("Cannot bind new neurons to parent layer. Neuron is not of type "
                        + IPropagateBack.class.getCanonicalName());
            }
        } else {
            throw new IllegalArgumentException("Your parent layer is output layer!");
        }
        return this;
    }

    /**
     * Fetch neurons inputs, sum them and update output to all neurons in here. Do it parallel, if you like.
     */
    public void propagate() {
        neurons.forEach(Neuron::propagate);
    }

    /**
     * Returns any output value in order of neurons
     */
    public float[] getOutputs() {
        float[] result = new float[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            result[i] = neuron.getOutputValue();
        }
        return result;
    }

    private void bindToLayerNeurons(Layer parentLayer, IPropagateBack n) {
        List<IBindableSourceNeuron> sourceNeurons = new ArrayList<>();
        for (Neuron neuron : parentLayer.neurons) {
            if (neuron instanceof IBindableSourceNeuron) {
                sourceNeurons.add((IBindableSourceNeuron) neuron);
            }
        }
        n.bindToInputNeurons(sourceNeurons);
    }

    /**
     * Identifies the building process to separate neuron types in class
     */
    private Function<Integer, Neuron> detectNeuronBuilder(NeuronType type, boolean softmax) {
        Function<Integer, Neuron> builder;

        switch (type) {
            case HIDDEN:
                builder = i -> new HiddenNeuron(format("{0}{1}/H{2}", type.toString(), name, String.valueOf(i)), this);
                break;
            case INPUT:
                builder = i -> new InputNeuron(format("{0}{1}/I{2}", type.toString(), name, String.valueOf(i)), this);
                break;
            case OUTPUT:
                builder = i -> new OutputNeuron(format("{0}{1}/O{2}", type.toString(), name, String.valueOf(i)), softmax, this);
                break;
            default:
                throw new IllegalArgumentException("Missing neuron type: " + type);
        }
        return builder;
    }

    /**
     * Returns the number of neurons in this layer
     */
    public int countNeurons() {
        return neurons.size();
    }

    /**
     * Returns weights to each neuron in this layer and it own weight to it input bindings
     */
    public List<List<Float>> getWeights() {
        return neurons.stream()
                .map(Neuron::getWeights)
                .collect(toList());
    }

    /**
     * Returns biases to each neuron
     */
    public List<Float> getBiases() {
        return neurons.stream()
                .map(Neuron::getBias)
                .collect(toList());
    }

  public static List<Neuron> getNeurons(Layer layer) {
        return layer.neurons;
    }

    /**
     * After every error value on each neuron was calculated, this function accepts each weight delta to its bindings.
     * This is the learning process. This annotation {@link LearningData} represents only a marker for learning relevant
     * data. It is a flag.
     */
    @LearningData
    void applyDeltas(float learningGradient) {
      this.neurons.forEach(childNeuron -> childNeuron.applyDelta(learningGradient));
    }

    /**
     * Sets the desired testing values to an output layer (only, else throw an exception).
     * After that, the error values will be set for every output neuron. Regarding hidden layer neurons.
     * This error will be used for parent layer neurons, this error need to be devided by weight effect.
     * Because multiple bound neurons connecting multiple neurons in a full mesh, this error is updated
     * multiple times. So the update error value and adjustment of weights need to be devided into two for-loops.
     *
     * @param desiredValues requires the exact same size as number of output neurons!
     */
    public void updateError(float[] desiredValues) {
        if (type != NeuronType.OUTPUT) {
            throw new RuntimeException("Cannot update desired values at another layer type than output layer! type = " + type);
        }

        if (desiredValues == null || desiredValues.length != countNeurons()) {
            throw new IllegalArgumentException("Your desired values must be count of neurons in the output layer!");
        }
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            neuron.setDesired(desiredValues[i]);
            neuron.updateError();
        }
    }

    /**
     * Updates each neuron error value for further parent layer neuron errors. Only callable for hidden layers
     */
    public void updateErrors() {
        if (type != NeuronType.HIDDEN) {
            throw new RuntimeException("Cannot run this method only for a hidden layer! type = " + type);
        }

        Layer childLayer = network.findChildLayer(this);
        if (childLayer == null) {
            throw new RuntimeException("No child layer found on layer: " + name);
        }
        for (Neuron parentNeuron : neurons) {
            parentNeuron.setError(childLayer.calculateHiddenError(parentNeuron));
        }
    }

  /**
   * Averages the differences between expectations and actual values
   */
  public float getTotalError(float[] desiredValues) {
    if (type != NeuronType.OUTPUT) {
      throw new RuntimeException("Can only be used on output layer!");
    }
    float sum = 0;
    for (int i = 0; i < neurons.size(); i++) {
      Neuron neuron = neurons.get(i);
      float desiredValue = desiredValues[i];
      float actualValue = neuron.getOutputValue();

      // need to be positive, because an error value can be fixed by increasing or descreasing
      sum = (float) Math.pow(desiredValue - actualValue, 2);
    }
    return sum / countNeurons();
  }

  public void setBiases(float bias) {
    neurons.forEach(n -> n.setBias(bias));
  }
}
