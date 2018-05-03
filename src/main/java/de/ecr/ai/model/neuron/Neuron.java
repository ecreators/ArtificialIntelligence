package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;
import de.ecr.ai.model.Layer;
import de.ecr.ai.model.annotation.LearningData;
import de.ecr.ai.model.neuron.activation.IActivationFunction;

import java.util.ArrayList;
import java.util.List;

import static java.util.stream.Collectors.toList;

/**
 * A neuron contain data representing the effect<br/>
 * of bound input neurons and their output values.<br/><br/>
 * <p>It can be classified as
 * <ul>
 * <li>input-neuron (no input bindings and fixed input values as output)</li>
 * <li>output-neuron (has input bindings and variable output values, based on bindings plus back propagation operations)</li>
 * <li>or as hidden-neuron (hast input binding and variable output values, based on bindings)</li>
 * </ul>
 * </p>
 *
 * @author Bjoern Frohberg
 */
public class Neuron {

    private float output;
    private float bias;
    private IActivationFunction activation = IActivationFunction.SIGMOID;
    final List<Binding> inputBindings;
    private final String name;
    private final Layer layer;
    private final NeuronType type;
    private float desired;
    private float error;

    protected Neuron(String name, Layer ownerLayer) {
        this.name = name;
        this.layer = ownerLayer;
        this.inputBindings = new ArrayList<>();

        if (this instanceof OutputNeuron) {
            type = NeuronType.OUTPUT;
        } else if (this instanceof InputNeuron) {
            type = NeuronType.INPUT;
        } else {
            type = NeuronType.HIDDEN;
        }
    }

    public final Layer getLayer() {
        return layer;
    }

    /**
     * Return a set output value
     */
    public final Float getOutputValue() {
        return output;
    }

    /**
     * Returns a set bias. A bias is a virtual offset to fix zero calculations.<br/>
     * It shifts the error calculation into a better range when calculating with zeros.
     */
    @LearningData
    public final float getBias() {
        return bias;
    }

    /**
     * Set a bias
     */
    public void setBias(float bias) {
        this.bias = bias;
    }

    /**
     * If required to change the activation function
     */
    public final void setActivation(IActivationFunction activation) {
        this.activation = activation;
    }

    /**
     * Simple identification for this neuron prototype
     */
    public final NeuronType getType() {
        return type;
    }

    /**
     * Update output value based on inputs
     */
    public void propagate() {

        // get sum from input x weight
        // add bias to sum
        // send through activation function
        // set output as value what comes from activation function

        double sum = inputBindings
                .stream()
                .mapToDouble(Binding::calculateOutput).sum();
        sum += bias;

        float inputSum = (float) sum;
        this.output = activation.activate(inputSum);
    }

    /**
     * Updates the output value to this neuron. Commonly this value is caluclated. <br/>
     * Only an input neuron directly sets this output as identity.
     */
    protected final void setOutput(float value) {
        this.output = value;
    }

    /**
     * Returns the calculated error
     */
    public final float getError() {
        return error;
    }

    /**
     * Set an error fixed
     */
    public final void setError(float error) {
        this.error = error;
    }

    /**
     * Returns any input bindings with weights
     */
    public final List<Binding> getInputBindings() {
        return inputBindings;
    }

    /**
     * Returns the neuron name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns number weight of each input bindings
     */
    public List<Float> getWeights() {
        return inputBindings.stream()
                .map(Binding::getWeight)
                .collect(toList());
    }

    /**
     * Sets a preferred value. Use for output layer.
     */
    public void setDesired(float desired) {
        this.desired = desired;
    }

    /**
     * Updates the error, output layer and hidden layer calculcation differ to each other.
     * This is only for use for the output layer, else throw an exception
     */
    public void updateError() {
        if (type != NeuronType.OUTPUT) {
            throw new RuntimeException("Cannot train another type than an output layer! " + type);
        }
        float outputValue = getOutputValue();
        setError(activation.derive(outputValue) * (desired - outputValue));
    }

    /**
     * Returns the parent neuron error by this child neuron in connection to
     */
    public float calculateParentError(Neuron parentNeuron) {
        float childNeuronInput = parentNeuron.getOutputValue();
        float derivated = activation.derive(childNeuronInput);
        float weight = getWeightIfParentNeuron(parentNeuron);
        float error = this.error;
        return derivated * weight * error;
    }

    /**
     * Get the weight value for a binding with the given parent neuron, else throw exception
     */
    private float getWeightIfParentNeuron(Neuron parentNeuron) {
        return inputBindings.stream()
                .filter(b -> b.isParentNeuron(parentNeuron))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Cannot find parent neuron!"))
                .getWeight();
    }

    /**
     * Adjusts bias and bindings weights by error delta
     */
    public void applyDelta(float learningGradient) {
        // updates the horizontal error shift, that fixes mostly local error minimum
        bias += learningGradient * error;

        // adjust weights
        inputBindings.forEach(b -> b.updateWeight(learningGradient, this));
    }
}
