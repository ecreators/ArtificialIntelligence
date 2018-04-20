package de.ecr.ai.model.neuron.activation;

/**
 * Gives you conversion for any value to put it between -1 and 1 (normalized ffrom a sum)
 *
 * @author Bjoern Frohberg
 */
public final class TangentHypActivation implements IActivationFunction {
	
	@Override
	public float activate(float sum) {
		return (float) Math.tanh(sum);
	}
}
