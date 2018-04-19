package de.ecr.ai.model.neuron.activation;

/**
 * @author Bjoern Frohberg
 */
public class SigmoidSoftMaxActivation implements IActivationFunction {
	
	private final float threshold;
	
	public SigmoidSoftMaxActivation(float threshold) {
		this.threshold = threshold;
	}
	
	@Override
	public float activate(float sum) {
		float sigmoid = IActivationFunction.SIGMOID.activate(sum);
		return sigmoid > threshold
		       ? 1
		       : 0;
	}
}
