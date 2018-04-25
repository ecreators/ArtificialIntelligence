package de.ecr.ai.model.neuron;

import java.util.List;

/**
 * Defines a interface to allow propagating an error delta backward up to its binding weights.<br/>
 * No adjustments to their input values!
 *
 * @author Bjoern Frohberg
 */
public interface IPropagateBack {
	
	/**
	 * Appends new Bindings for given neurons
	 */
  void bindToInputNeurons(List<IBindableSourceNeuron> neurons);

}
