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
public abstract class Neuron {

    private float output;
    private float bias;
    private IActivationFunction activation = IActivationFunction.SIGMOID;
    final List<Binding> inputBindings;
    private final String name;
    private final Layer layer;
    private final NeuronType type;

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
    public final float getOutputValue() {
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
     * Returns an activation function
     */
    public final IActivationFunction getActivation() {
        return activation;
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
     * Calculates an error value pending on child neuron output values and their depending input bindings
     */
    protected abstract float calculateError();

    /**
     * Updates the output value to this neuron. Commonly this value is caluclated. <br/>
     * Only an input neuron directly sets this output as identity.
     */
    protected final void setOutput(float value) {
        this.output = value;
    }

    /**
     * Returns the calculated error
     *
     * @deprecated Will be removed. Use {@link #calculateError()} instead. (#3)
     */
    @Deprecated
    public final float getError() {
        return calculateError();
    }

    /**
     * Set an error fixed
     *
     * @deprecated Will be removed. Dies nothing right now (#3)
     */
    @Deprecated
    public final void setError(float error) {
        // TODO removed
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
}
