package de.ecr.ai.model.neuron;

/**
 * Determine a neuron inside a hidden layer.
 * This neuron has variable input values and propagate a value forward and backward.
 * A hidden neuron is a complexity for the neural network to do a fine tuning in its
 * strategy to solve an output value. // TODO see "hidden" layers
 *
 * @author Bjoern Frohberg
 */
public final class HiddenNeuron extends Neuron implements IPropagateBack {
	
	@Override
	protected float getError() {
		
		// TODO calculate error based on output bindings for this neuron
		
		// An error value in a hidden layer represents an fine adjustment
		// based on adjusted output binding weights. Will be clear after implementation.
		// Just for known, this error is a balanced adjustment pending on output values.
		// This is because adjustments to its weights, so we cannot use a fully values as
		// in the output layer, because we will have multiple output value to regard.
		
		return 0;
	}
	
	@Override
	public void propagateBackward(float learningGradient) {
	
		// Need to use getError to update input binding weights
		
		// TODO implement
	}
}
