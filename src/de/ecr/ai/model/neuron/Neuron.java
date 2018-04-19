package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;
import de.ecr.ai.model.neuron.activation.IActivationFunction;

import java.util.ArrayList;
import java.util.List;

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
	
	private         float               output;
	private         float               bias;
	//	private         float               errorDelta;
	private         IActivationFunction activation = IActivationFunction.SIGMOID;
	// TODO reference layer : Layer
	protected final List<Binding>       inputBindings;
	private final   String              name;
	
	protected Neuron(String name) {
		this.name = name;
		inputBindings = new ArrayList<>();
	}
	
	/**
	 * Return a set output value
	 */
	public final float getOutputValue() {
		return output;
	}
	
	/**
	 * Returns a set bias
	 */
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
		if(this instanceof OutputNeuron) {
			return NeuronType.OUTPUT;
		}
		if(this instanceof InputNeuron) {
			return NeuronType.INPUT;
		}
		return NeuronType.HIDDEN;
	}
	
	/**
	 * Update output value based on inputs
	 *
	 * @deprecated Will be removed and handled outside (#3)
	 */
	@Deprecated
	public void propate() {
//		NeuronType type = getType();
//		if(type == NeuronType.INPUT) {
//			return;
//		}
//
//		float inputSum = (float) (inputBindings.parallelStream().mapToDouble(Binding::calculateOutput).sum() + bias);
//		this.output = activation.activate(inputSum);
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
	protected final List<Binding> getInputBindings() {
		return inputBindings;
	}
	
	/**
	 * Returns the neuron name
	 */
	public String getName() {
		return name;
	}
}
