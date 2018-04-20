package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;

import java.util.List;

/**
 * Defines a fassade to allow propagating an error delta backward up to its binding weights.<br/>
 * No adjustments to their input values!
 *
 * @author Bjoern Frohberg
 */
public interface IPropagateBack {
	
	/**
	 * Appends new Bindings for given neurons
	 */
	List<Binding> bindToInputNeurons(List<IBindableSourceNeuron> neurons);
	
	/**
	 * Train delta for the neuron bias
	 */
	default void learn(float learningGradient) {
		
		// only output and hidden neurons
		
		Neuron n = (Neuron) this;
		
		final float error = n.calculateError();
		
		n.setBias(n.getBias() + error * learningGradient);
		
		// binding: Weight += Maths.CalculateWeightDelta(childNeuron, ParentNeuron.Output, learningRate);
		
		n.getInputBindings().parallelStream().forEach(b -> b.learn(learningGradient, error));
	}
}
