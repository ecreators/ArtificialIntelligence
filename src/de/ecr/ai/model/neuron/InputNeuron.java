package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Layer;
import de.ecr.ai.model.exception.NotImplementedException;

import static java.text.MessageFormat.format;

/**
 * Determine this neuron as fixed neuron with fixed input values.<br/>
 * The value is set from the outside and must be normalized between 0f and 1f.
 *
 * @author Bjoern Frohberg
 */
public final class InputNeuron extends Neuron implements IBindableSourceNeuron {
	
	public InputNeuron(String name, Layer ownerLayer) {
		super(name, ownerLayer);
	}
	
	/**
	 * Allows to set the output directly as an input value for this input neuron.
	 *
	 * @throws IllegalArgumentException Your value need to be valid between 0 (zero) and 1 (one)
	 */
	public void setInputValue(float value) throws IllegalArgumentException {
		
		if(value < 0 || value > 1) {
			throw new IllegalArgumentException(format("Your value '{0}' is not normalized between 0 and 1!", value));
		}
		
		super.setOutput(value);
	}
	
	@Override
	protected float calculateError() throws RuntimeException {
		throw new NotImplementedException("Cannot use error for input neuron! Input values won't be update!");
	}
}
