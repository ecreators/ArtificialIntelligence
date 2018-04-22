package de.ecr.ai.model.test;

import de.ecr.ai.model.Layer;
import de.ecr.ai.model.NeuralNetwork;

/**
 * A unit to pass input values and desired values during a {@link TrainingSession}.
 * TODO TrainingSession missing to this time, comes with future comments
 *
 * @author Bjoern Frohberg
 */
public final class TestUnit {
	
	/**
	 * Input Values for the Input-{@link Layer}
	 */
	public float[] inputValues;
	
	/**
	 * Desired output values for the training-method to train the {@link NeuralNetwork}
	 */
	public float[] desiredValues;
	
	// HINT: you can extend this attributes for debug informations, may be for visualization ...
}
