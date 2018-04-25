package de.ecr.ai.model.neuron;

import de.ecr.ai.model.Binding;
import de.ecr.ai.model.Layer;

import java.util.List;
import java.util.stream.Collectors;

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
  public void bindToInputNeurons(List<IBindableSourceNeuron> neurons) {
		inputBindings.addAll(neurons.stream().map(n -> new Binding(this, n)).collect(Collectors.toList()));
  }
}
