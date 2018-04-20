package de.ecr.ai.model.neuron.activation;

/**
 * Defines a strategy to put a sum value to a normalized value between optimized range
 *
 * @author Bjoern Frohberg
 */
public interface IActivationFunction {
	
	/**
	 * {@link SigmoidActivation}
	 */
	IActivationFunction SIGMOID = new SigmoidActivation();
	
	/**
	 * {@link TangentHypActivation}
	 */
	IActivationFunction TANGENT_HYPERBOLIC = new TangentHypActivation();
	
	/**
	 * Convert a sum to a normalized value
	 */
	float activate(float sum);
	
	/**
	 * Inverse a normalized value to a normalized input value
	 */
	default float derive(float output) {
		return output * (1 - output);
	}
}
