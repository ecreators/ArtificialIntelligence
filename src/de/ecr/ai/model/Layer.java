package de.ecr.ai.model;

import de.ecr.ai.model.neuron.*;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.text.MessageFormat.format;

/**
 * Defines a cluster to neurons to align a related solution path for the neural network
 *
 * @author Bjoern Frohberg
 */
public final class Layer {
	
	/*
	Path to go structure
	
	create layer Input, then Hidden, may be a second Hidden, last Output
	clear neurons <-- if restored, else not nessessary
	build neurons
	bind full mesh
	
	-- ready to go
	
	call propagate
	call getOutputValues for test (in test AND training)
	
	(only training)
	TODO call train set desired values for output
	TODO in TEST-Method handle, when to stop iteration, if so doing
	 */
	
	private final String       name;
	private final List<Neuron> neurons;
	private       NeuronType   type;
	// TODO implement Neural Network class first!
	
	public Layer(String name) {
		this.name = name;
		this.neurons = new ArrayList<>();
	}
	
	/**
	 * Empties neurons list
	 */
	public void clear() {
		this.neurons.clear();
	}
	
	/**
	 * Appends new neurons (no softmax on output)
	 */
	public Layer createNeurons(int neuronsCount, NeuronType type, boolean softmax) {
		this.type = type;
		
		Function<Integer, Neuron> builder = detectNeuronBuilder(type, softmax);
		
		IntStream.range(0, neuronsCount).parallel().mapToObj(builder::apply).forEach(this.neurons::add);
		
		// to append bindFullMesh
		return this;
	}
	
	/**
	 * Returns the layer neuron type used to create neurons on this layer
	 */
	public NeuronType getType() {
		return type;
	}
	
	/**
	 * Bind any neuron in this layer to each neuron in the parent layer neurons as a fully mesh.
	 * Requires a layer of type {@link NeuronType#INPUT} or {@link NeuronType#HIDDEN}
	 */
	public void bindFullMesh(Layer parentLayer) {
		if(type != NeuronType.OUTPUT) {
			neurons.parallelStream()
			       .map(n -> (IPropagateBack) n)
			       .forEach(n -> bindToLayerNeurons(parentLayer, n));
		} else {
			throw new IllegalArgumentException("Your parent layer is output layer!");
		}
	}
	
	/**
	 * Fetch neurons inputs, sum them and update output to all neurons in here. Do it parallel, if you like.
	 */
	public void propagate() {
		neurons.parallelStream().forEach(Neuron::propagate);
	}
	
	/**
	 * Returns any output value in order of neurons
	 */
	public float[] getOutputs() {
		float[] result = new float[neurons.size()];
		for (int i = 0; i < neurons.size(); i++) {
			Neuron neuron = neurons.get(i);
			result[i] = neuron.getOutputValue();
		}
		return result;
	}
	
	private void bindToLayerNeurons(Layer parentLayer, IPropagateBack n) {
		List<IBindableSourceNeuron> sourceNeurons = parentLayer.neurons.parallelStream()
		                                                               .map(p -> (IBindableSourceNeuron) p)
		                                                               .collect(Collectors.toList());
		n.bindToInputNeurons(sourceNeurons);
	}
	
	/**
	 * Identifies the building process to separate neuron types in class
	 */
	private Function<Integer, Neuron> detectNeuronBuilder(NeuronType type, boolean softmax) {
		Function<Integer, Neuron> builder;
		
		switch (type) {
			case HIDDEN:
				builder = i -> new HiddenNeuron(format("{0}{1}/H{2}", type.toString(), name, String.valueOf(i)), this);
				break;
			case INPUT:
				builder = i -> new InputNeuron(format("{0}{1}/I{2}", type.toString(), name, String.valueOf(i)), this);
				break;
			case OUTPUT:
				builder = i -> new OutputNeuron(format("{0}{1}/O{2}", type.toString(), name, String.valueOf(i)), softmax, this);
				break;
			default:
				throw new IllegalArgumentException("Missing neuron type: " + type);
		}
		return builder;
	}
}
