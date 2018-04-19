package de.ecr.ai.model.neuron;

/**
 * Determine this neuron as output. <br/>
 * A layer propagation after this layer neurons won't update further child layer neurons, <br/>
 * there is no further child layer with neurons.
 *
 * @author Bjoern Frohberg
 */
public final class OutputNeuron extends Neuron implements IPropagateBack {
	
	private float desired;
	
	@Override
	protected float getError() {
		
		// TODO calculate using bindings -> implement bindings first
		
		// activate sum of each value of input binding in times binding weight and plus bias of this neurons
		// error value is delta between predicted ourput value given by activation and desired output value,
		// only during training
		
		return 0;
	}
	
	@Override
	public void propagateBackward(float learningGradient) {
		
		// Need to use getError to update input binding weights
		
		// TODO implement
	}
	
	/**
	 * A simple setter for desired output value
	 */
	public float getDesiredValue() {
		return desired;
	}
	
	/**
	 * Returns the set desired output value, if so done
	 */
	public void setDesired(float value) {
		this.desired = value;
	}
}
