package de.ecr.ai.model.neuron.activation;

import de.ecr.ai.model.neuron.Neuron;
import de.ecr.ai.model.neuron.OutputNeuron;

/**
 * Use for a classificational result.
 * This is a result with multiple outputs to get only one output near by or exact 1 and every output else 0.
 *
 * @author Bjoern Frohberg
 */
public class SoftMaxActivation implements IActivationFunction {
	
	private final Neuron neuron;
	
	public SoftMaxActivation(OutputNeuron n) {
		this.neuron = n;
	}
	
	@Override
	public float activate(float sum) {
		
		// calculates a
		
		
		// float total = (float)n.getLayer().getNeurons().stream().map(n -> Math.exp(n.getOutput())).sum();
		// TODO implement layer first! (#3)
		float total = 1f;
		
		return (float) (Math.exp(sum) / total);
	}
}
