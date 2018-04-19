package de.ecr.ai.model.neuron;

/**
 * Defines a fassade to allow propagating an error delta backward up to its binding weights.<br/>
 * No adjustments to their input values!
 *
 * @author Bjoern Frohberg
 */
public interface IPropagateBack {
	
	/**
	 * Push an error delta weighted to the input bindings
	 */
	void propagateBackward(float learningGradient);
}
