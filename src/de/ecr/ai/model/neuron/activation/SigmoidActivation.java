package de.ecr.ai.model.neuron.activation;

/**
 * Gives you conversion for any value to put it between 0 and one (normalized from a sum)
 *
 * @author Bjoern Frohberg
 */
public final class SigmoidActivation implements IActivationFunction {
	
	@Override
	public float activate(float sum) {
		return (float) (1d / (1d + Math.exp(-sum)));
	}
}
