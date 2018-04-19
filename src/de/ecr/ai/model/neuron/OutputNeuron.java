package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;
import de.ecr.ai.model.neuron.activation.SoftMaxActivation;

import java.util.List;

/**
 * Determine this neuron as output. <br/>
 * A layer propagation after this layer neurons won't update further child layer neurons, <br/>
 * there is no further child layer with neurons.
 *
 * @author Bjoern Frohberg
 */
public final class OutputNeuron extends Neuron implements IPropagateBack {
	
	private float desired;
	
	/**
	 * Defines the output tolerant between 0 and 1 (inclusive)
	 */
	public OutputNeuron(String name) {
		this(name, false);
	}
	
	/**
	 * Sets a softmax sigmoid activation function, if not {@code null}.
	 *
	 * @param softmax null or a value between 0 and 1 to maximize the result to a boolean 0 or 1
	 */
	public OutputNeuron(String name, boolean softmax) {
		super(name);
		if(softmax) {
			setActivation(new SoftMaxActivation(this));
		}
	}
	
	@Override
	protected float calculateError() {
		
		// activate sum of each value of input binding in times binding weight and plus bias of this neurons
		// error value is delta between predicted ourput value given by activation and desired output value,
		// only during training
		
		float des        = getDesiredValue();
		float out        = getOutputValue();
		float errorDelta = des - out;
		float derive     = getActivation().derive(out);
		
		return derive * errorDelta;
	}
	
	@Override
	public List<Binding> bindToInputNeurons(List<IBindableSourceNeuron> neurons) {
		neurons.parallelStream().map(n -> new Binding(this, n)).forEach(inputBindings::add);
		return inputBindings;
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
