package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;
import de.ecr.ai.model.Layer;
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
	public OutputNeuron(String name, Layer ownerLayer) {
		this(name, false, ownerLayer);
	}
	
	/**
	 * Sets a softmax sigmoid activation function, if not {@code null}.
	 *
	 * @param softmax null or a value between 0 and 1 to maximize the result to a boolean 0 or 1
	 */
	public OutputNeuron(String name, boolean softmax, Layer ownerLayer) {
		super(name, ownerLayer);
		if(softmax) {
			setActivation(new SoftMaxActivation(this));
		}
	}

	@Override
  public void bindToInputNeurons(List<IBindableSourceNeuron> neurons) {
		neurons.parallelStream().map(n -> new Binding(this, n)).forEach(inputBindings::add);
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
