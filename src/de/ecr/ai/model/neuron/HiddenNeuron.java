package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;
import de.ecr.ai.model.Layer;

import java.util.List;

/**
 * Determine a neuron inside a hidden layer.
 * This neuron has variable input values and propagate a value forward and backward.
 * A hidden neuron is a complexity for the neural network to do a fine tuning in its
 * strategy to solve an output value. // TODO see "hidden" layers
 *
 * @author Bjoern Frohberg
 */
public final class HiddenNeuron extends Neuron implements IPropagateBack, IBindableSourceNeuron {
	
	public HiddenNeuron(String name, Layer ownerLayer) {
		super(name, ownerLayer);
	}
	
	@Override
	protected float calculateError() {
		
		// TODO calculate error based on output bindings for this neuron
		
		// An error value in a hidden layer represents an fine adjustment
		// based on adjusted output binding weights. Will be clear after implementation.
		// Just for known, this error is a balanced adjustment pending on output values.
		// This is because adjustments to its weights, so we cannot use a fully values as
		// in the output layer, because we will have multiple output value to regard.
		
		/* HELP - preview
		Maths.CalculateDeltaHidden(pNeuron.Output, childNeuron.Error, childNeuron.Bindings[pNeuron.ID].Weight)
		= Derivate(inputValue) * outputBindingWeight * outputBindingDelta
		
		parentNeuron.Error = childLayer.Neurons.Sum(childNeuron => Maths.CalculateDeltaHidden(parentNeuron.Output, childNeuron.Error, childNeuron.Bindings[parentNeuron.ID].Weight));
		*/
		
		return 0;
	}
	
	@Override
	public List<Binding> bindToInputNeurons(List<IBindableSourceNeuron> neurons) {
		neurons.parallelStream().map(n -> new Binding(this, n)).forEach(inputBindings::add);
		return inputBindings;
	}
}
