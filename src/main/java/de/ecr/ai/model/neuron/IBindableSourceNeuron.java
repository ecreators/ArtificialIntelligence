package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;

/**
 * Declares a neuron usable for {@link Binding}
 *
 * @author Bjoern Frohberg
 */
public interface IBindableSourceNeuron {
	
	/**
	 * Returns the calculated output value
	 */
	float getOutputValue();
	
	/**
	 * Returns neuron name
	 */
	String getName();
}
