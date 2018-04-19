package de.ecr.ai.model.neuron;

import de.ecr.ai.model.neuron.activation.IActivationFunction;

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
	
	private float               output;
	private float               bias;
	private float               errorDelta;
	private IActivationFunction activation = IActivationFunction.SIGMOID;
	// TODO layer : Layer
	// TODO bindings : Binding[]
	
	public final float getOutputValue() {
		return output;
	}
	
	public final float getBias() {
		return bias;
	}
	
	public void setBias(float bias) {
		this.bias = bias;
	}
	
	public final IActivationFunction getActivation() {
		return activation;
	}
	
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
	 */
	public void propate() {
		if(getType() == NeuronType.INPUT) {
			//noinspection UnnecessaryReturnStatement
			return;
		}
		
		// TODO layer, bindings first
	}
	
	public void learn(float gradient) {
		// TODO layer, bindings first
		
		float outputDeltaError = getError();
	}
	
	protected abstract float getError();
	
	protected final void setOutput(float value) {
		this.output = value;
	}
	
	public final float getErrorDelta() {
		return errorDelta;
	}
	
	public final void setErrorDelta(float errorDelta) {
		this.errorDelta = errorDelta;
	}
}
