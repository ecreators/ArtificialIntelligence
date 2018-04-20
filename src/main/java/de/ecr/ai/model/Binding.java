package de.ecr.ai.model;

import de.ecr.ai.model.neuron.IBindableSourceNeuron;
import de.ecr.ai.model.neuron.Neuron;

import java.security.SecureRandom;
import java.util.Random;

import static java.text.MessageFormat.format;

/**
 * Binding to a parent neuron (neuron in previous layer [hidden layer or input layer]
 *
 * @author Bjoern Frohberg
 */
public final class Binding {
	
	private static final Random RANDOM = new SecureRandom();
	
	private final IBindableSourceNeuron parentNeuron;
	private final String                childName;
	private       float                 weight;
	
	public Binding(Neuron child, IBindableSourceNeuron parentNeuron) {
		this.parentNeuron = parentNeuron;
		this.childName = child.getName();
		randomizeWeight();
	}
	
	/**
	 * For visualization
	 */
	public String getName() {
		return format("{0}:-( {1} )->:{2}", parentNeuron.getName(), weight, childName);
	}
	
	/**
	 * Initial weight between -1 and 1
	 */
	public void randomizeWeight() {
		weight = RANDOM.nextFloat() * 2 - 1;
	}
	
	/**
	 * Use it during loading a neural network
	 */
	public void setWeight(float weight) {
		this.weight = weight;
	}
	
	/**
	 * Returns a calculated balanced value depending on the parent neuron output value
	 */
	public float calculateOutput() {
		return weight * parentNeuron.getOutputValue();
	}
	
	public void learn(float learningGradient, float error) {
		float in = parentNeuron.getOutputValue();
		weight += learningGradient * in * error;
	}
}
