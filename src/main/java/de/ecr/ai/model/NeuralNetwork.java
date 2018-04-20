package de.ecr.ai.model;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.ecr.ai.model.exception.NotImplementedException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.text.MessageFormat.format;
import static java.util.stream.Collectors.toList;

/**
 * The neural network is our connectivity to the brain (weights and biases). <br/>
 * The class {@link NeuralNetwork} is a hull for nessessary technics a neural network must be to do.
 *
 * @author Bjoern Frohberg
 */
public final class NeuralNetwork {
	
	private final List<Layer> layers;
	private       String      name;
	private       float       learningGradient = 0.15f;
	private       long        generations; // to check version of the network
	
	public NeuralNetwork() {
		this.layers = new ArrayList<Layer>();
	}
	
	/**
	 * Returns the leaning gradient the network will close up to expected output values,<br/>
	 * during a training session in back pass (or back propagation).<br/>
	 * This value should be between 0 (no learning process) and 1 (no evolution).<br/>
	 * Only evolution process will teach the network to find a way through training sessions,<br/>
	 * to identify a solution and a concept between your input and (expected) output values.<br/>
	 * <br/>
	 * <b>default is 0.15 or 0.65 (recommened is an odd value)</b>
	 */
	public float getLearningGradient() {
		// TODO set leaningGradient in preparation for training session.
		// TODO training session data first!
		return learningGradient;
	}
	
	/**
	 * Propagate new or learned input values (normalized between 0 and 1) through a (un)trained
	 * neural network and passes the predicted output values back as result.
	 */
	public float[] test(float... inputValues) {
		
		// TODO comes in another commit
		
		throw new NotImplementedException("need to be implemented!");
	}
	
	/**
	 * Describes this network. This method will be used when using load-methods.
	 * But you can use them manually as well for testing.
	 *
	 * @param inputs            How many input values do you need?
	 *                          Think about: what will value affects possibly your expected outputs?
	 * @param hiddenLayersCount How slow and complex will you create your network?<br/>
	 *                          Think about: fewer hidden layers speed up your results, but more layers<br/>
	 *                          will give your network more ways to identify the "perfect way", but on<br/>
	 *                          performance costs and with a greater chance to skip the global minimum.<br/>
	 *                          <b>The global minimum describes a moment, when each test case of your<br/>
	 *                          session results in the lowest error delta you can get.</b><br/>
	 *                          Mostly your network will gonna be tricked and gives you local minima, but<br/>
	 *                          pending on the learning gradient this problem can be fixed. May be you<br/>
	 *                          need to adjust the learning gradient a little bit higher. But remember, if<br/>
	 *                          your learning gradient is too high you propably never find the global<br/>
	 *                          minimum. This can happen, when your network is at a point, when the range<br/>
	 *                          (learning gradient) is bigger than the range where the global minimum is<br/>
	 *                          aligned. The global minimum describes a range where every situation of your<br/>
	 *                          use / test cases can be predicted with the highest correctness.<br/>
	 *                          You can identify if your network stucks in a local minimum, if you visualize<br/>
	 *                          the error delta total for a longer time. A neural network should be able<br/>
	 *                          to reach 95 to 100 % correctness, meaning a total error of 0 to 5 %. If it<br/>
	 *                          stuck on 20, may be 30 percent, then your network should be modify the<br/>
	 *                          random initial weights or only some of them and restart the training or adjust<br/>
	 *                          the learning gradient a few.
	 * @param hiddenNeurons     This neurons describe a pattern matcher algorithm. The network will tweak<br/>
	 *                          the weights	and have some space to act and learn on this weights for it self.<br/>
	 *                          If they are too few, it is like a pupil with 2 pencils of red and green and<br/>
	 *                          a request to draw a blue line. But don't give your pupil not a buldozer to<br/>
	 *                          draw a blue line. Meaning given him to less and your network won't find always<br/>
	 *                          the correct outputs. Too much and your network gets slow and fails more than<br/>
	 *                          learn. You can draw a blue line with a bulldozer, but it's difficult to identify<br/>
	 *                          how.
	 * @param outputNeurons     What will your network control? Think about: Your network has a answer to respond.<br/>
	 *                          The meaning of the neural network is to predict at least one output. Your output<br/>
	 *                          always will be a floating number between 0 and 1 (inclusive). Find out, what you<br/>
	 *                          want to identify and how the network should be able to identify this result.<br/>
	 *                          If you want your network should answer yes or no, the network should have one output<br/>
	 *                          with rounding to it's prediction to get 0 or 1. If you want to do some<br/>
	 *                          classification, your network should cover each option with an own output value, such<br/>
	 *                          as letters from "a" to "z". You can also set a number for descissions the network<br/>
	 *                          should choose. Your network will learn, how to use the input values and balance it<br/>
	 *                          to reach your training data. If the network is fitting the outputs, then your<br/>
	 *                          network is ready for unknown input values and it will be good to predict their<br/>
	 *                          output values close correctly.
	 * @param outputSoftmax     Will you need to get only 1 descission on many outputs to be set to 1 and every else<br/>
	 *                          output to be zero then? then use {@code true} for this value.
	 */
	public void build(int inputs, int hiddenLayersCount, int hiddenNeurons, int outputNeurons, boolean outputSoftmax) {
		
		// TODO comes with another commit ;-)
		
		throw new NotImplementedException("need to be implemented!");
	}
	
	/**
	 * Rebuilds your neural network based on json data for {@link MemoryData}.
	 * If your json is valid, method {@link #loadMemory(MemoryData)} will be called.
	 */
	public void loadMemoryDataFromJson(String jsonMemoryData) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			Class<MemoryData> targetType = MemoryData.class;
			MemoryData        memoryData = mapper.readValue(jsonMemoryData, targetType);
			if(memoryData == null) {
				throw new IllegalArgumentException(format("cannot read {0} from json argument!",
				                                          targetType.getCanonicalName()));
			}
			loadMemory(memoryData);
		} catch (IOException ex) {
			throw new RuntimeException(ex);
		}
	}
	
	/**
	 * Rebuilds yourneural network based on memory data (propably from a file)
	 */
	public void loadMemory(MemoryData data) {
		this.layers.clear();
		build(data.inputs, data.hiddenLayers, data.hiddenNeurons, data.outputs, data.softMaxUsed);
		bindLayerNeuronsFullMesh(this.layers);
	}
	
	/**
	 * Collects memory data to store them external
	 */
	public MemoryData readMemory() {
		MemoryData data = new MemoryData();
		data.inputs = layers.get(0).countNeurons();
		data.outputs = layers.get(layers.size() - 1).countNeurons();
		data.hiddenLayers = layers.size() - 2;
		data.hiddenNeurons = layers.stream()
		                           .skip(1).limit(data.hiddenLayers)
		                           .mapToInt(Layer::countNeurons)
		                           .sum();
		
		data.weights = layers.stream().map(Layer::getWeights).collect(toList());
		data.biases = layers.stream().map(Layer::getBiases).collect(toList());
		data.networkName = getName();
		data.learningGradient = learningGradient;
		data.generations = generations;
		return data;
	}
	
	/**
	 * Returns a custom name to identify the network use cases
	 */
	public final String getName() {
		return name;
	}
	
	/**
	 * Describes the neural network target operation
	 */
	public void setName(String name) {
		this.name = name;
	}
	
	/**
	 * Converts {@link MemoryData} from {@link #readMemory()} into a json string
	 */
	public String saveToJson() {
		return saveToJson(readMemory());
	}
	
	/**
	 * Converts {@link MemoryData} from {@link #readMemory()} into a json string
	 */
	public String saveToJson(MemoryData data) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			return mapper.writeValueAsString(data);
		} catch (JsonProcessingException ex) {
			throw new RuntimeException(ex);
		}
	}
	
	/**
	 * Meshes the entire neural network layer neurons to parent layer neurons as full-mesh
	 */
	private void bindLayerNeuronsFullMesh(List<Layer> layers) {
		
		// TODO comes with a future commit
		
		throw new NotImplementedException("not yet implemented: bind every neuron in a mesh");
	}
	
	// TODO how do a training session? prototype TrainingSession and TrainingUnit
}
