package de.ecr.ai.model;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import de.ecr.ai.model.annotation.ForTest;
import de.ecr.ai.model.neuron.InputNeuron;
import de.ecr.ai.model.neuron.Neuron;
import de.ecr.ai.model.neuron.NeuronType;
import de.ecr.ai.model.test.TestUnit;
import de.ecr.ai.model.test.TrainingSession;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static java.text.MessageFormat.format;
import static java.util.stream.Collectors.toList;

/**
 * The neural network is our connectivity to the brain (weights and biases). <br/>
 * The class {@link NeuralNetwork} is a hull for nessessary technics a neural network must be to do.
 *
 * @author Bjoern Frohberg
 */
@SuppressWarnings("WeakerAccess")
public final class NeuralNetwork {

  /**
   * Don't stop after an amount of generations
   * See {@link #evolute(int, TrainingSession, float)}
   */
  public static final int GENERATIONS_MAX = 0;

  private final List<Layer> layers;
  private String name;
  private float learningGradient = 0.15f;
  private long generations; // to check version of the network
  private float totalError;
  private float biasAll;

  public NeuralNetwork() {
    this.layers = new ArrayList<>();
  }

  /**
   * Returns the current training generation session total error. Quality = 1 - totalError.
   * Is only representation how effective your network will be.
   */
  @SuppressWarnings("unused")
  public float getTotalError() {
    return totalError;
  }

  /**
   * Returns the leaning gradient the network will close up to expected output values,<br/>
   * during a training session in back pass (or back propagation).<br/>
   * This value should be between 0 (no learning process) and 1 (no evolution).<br/>
   * Only evolution process will teach the network to find a way through training sessions,<br/>
   * to identify a solution and a concept between your input and (expected) output values.<br/>
   * <br/>
   * <b>default is 0.15 or 0.65 (recommend is an odd value)</b>
   */
  @SuppressWarnings("unused")
  public float getLearningGradient() {
    // set leaningGradient in preparation for training session.
    // training session data first!
    return learningGradient;
  }

  /**
   * Propagate new or learned input values (normalized between 0 and 1) through a (un)trained
   * neural network and passes the predicted output values back as result.
   */
  public Float[] test(float... inputValues) {

    Layer inputLayer = layers.get(0);
    if (inputValues == null || inputValues.length != inputLayer.countNeurons()) {
      throw new IllegalArgumentException("inputValues must be the same count");
    }

    List<Neuron> neurons = Layer.getNeurons(inputLayer);
    for (int i = 0; i < neurons.size(); i++) {
      InputNeuron neuron = (InputNeuron) neurons.get(i);
      neuron.setInputValue(inputValues[i]);
    }
    layers.stream()
      .filter(l -> l.getType() != NeuronType.INPUT)
      .forEach(Layer::propagate);

    int lastLayerInput = layers.size() - 1;
    return layers.get(lastLayerInput).getOutputs();
  }

  /**
   * Describes this network. This method will be used when using load-methods.
   * But you can use them manually as well for testing.
   *
   * @param inputs            How many input values do you need?
   *                          Think about: what will value affects possibly your expected outputs?
   * @param hiddenLayersCount - How slow and complex will you create your network?
   *                          Think about: fewer hidden layers speed up your results, but more layers
   *                          will give your network more ways to identify the "perfect way", but on
   *                          performance costs and with a greater chance to skip the global minimum.
   *                          <b>The global minimum describes a moment, when each test case of your
   *                          session results in the lowest error delta you can get.</b>
   *                          Mostly your network will gonna be tricked and gives you local minima, but
   *                          pending on the learning gradient this problem can be fixed. May be you
   *                          need to adjust the learning gradient a little bit higher. But remember, if
   *                          your learning gradient is too high you propably never find the global
   *                          minimum. This can happen, when your network is at a point, when the range
   *                          (learning gradient) is bigger than the range where the global minimum is
   *                          aligned. The global minimum describes a range where every situation of your
   *                          use / test cases can be predicted with the highest correctness.
   *                          You can identify if your network stucks in a local minimum, if you visualize
   *                          the error delta total for a longer time. A neural network should be able
   *                          to reach 95 to 100 % correctness, meaning a total error of 0 to 5 %. If it
   *                          stuck on 20, may be 30 percent, then your network should be modify the
   *                          random initial weights or only some of them and restart the training or adjust
   *                          the learning gradient a few.
   * @param hiddenNeurons     This neurons describe a pattern matcher algorithm. The network will tweak
   *                          the weights	and have some space to act and learn on this weights for it self.
   *                          If they are too few, it is like a pupil with 2 pencils of red and green and
   *                          a request to draw a blue line. But don't give your pupil not a buldozer to
   *                          draw a blue line. Meaning given him to less and your network won't find always
   *                          the correct outputs. Too much and your network gets slow and fails more than
   *                          learn. You can draw a blue line with a bulldozer, but it's difficult to identify
   *                          how.
   * @param outputNeurons     What will your network control? Think about: Your network has a answer to respond.
   *                          The meaning of the neural network is to predict at least one output. Your output
   *                          always will be a floating number between 0 and 1 (inclusive). Find out, what you
   *                          want to identify and how the network should be able to identify this result.
   *                          If you want your network should answer yes or no, the network should have one output
   *                          with rounding to it's prediction to get 0 or 1. If you want to do some
   *                          classification, your network should cover each option with an own output value, such
   *                          as letters from "a" to "z". You can also set a number for descissions the network
   *                          should choose. Your network will learn, how to use the input values and balance it
   *                          to reach your training data. If the network is fitting the outputs, then your
   *                          network is ready for unknown input values and it will be good to predict their
   *                          output values close correctly.
   * @param outputSoftmax     Will you need to get only 1 descission on many outputs to be set to 1 and every else<br/>
   *                          output to be zero then? then use {@code true} for this value.
   */
  public void build(int inputs, int hiddenLayersCount, int hiddenNeurons, int outputNeurons, boolean outputSoftmax) {

    Layer parentLayer = createLayer("input", inputs, NeuronType.INPUT, false);
    for (int i = 0; i < hiddenLayersCount; i++) {
      parentLayer = createLayer("hidden", hiddenNeurons, NeuronType.HIDDEN, false).bindFullMesh(parentLayer);
    }
    createLayer("output", outputNeurons, NeuronType.OUTPUT, outputSoftmax).bindFullMesh(parentLayer);

    layers.forEach(l -> l.setBiases(biasAll));
  }

  private Layer createLayer(String name, int neuronCount, NeuronType layerType, boolean outputSoftmax) {
    Layer layer = new Layer(name, this);
    layers.add(layer);
    layer.createNeurons(neuronCount, layerType, outputSoftmax);
    return layer;
  }

  /**
   * Rebuilds your neural network based on json data for {@link MemoryData}.
   * If your json is valid, method {@link #loadMemory(MemoryData)} will be called.
   */
  public void loadMemoryDataFromJson(String jsonMemoryData) {
    ObjectMapper mapper = new ObjectMapper();
    try {
      Class<MemoryData> targetType = MemoryData.class;
      MemoryData memoryData = mapper.readValue(jsonMemoryData, targetType);
      if (memoryData == null) {
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
    this.generations = data.generations;
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
  @SuppressWarnings("unused")
  public void setName(String name) {
    this.name = name;
  }

  /**
   * Converts {@link MemoryData} from {@link #readMemory()} into a json string
   */
  @SuppressWarnings("unused")
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
    Layer parentLayer = null;
    for (Layer layer : layers) {
      if (parentLayer != null) {
        parentLayer = layer.bindFullMesh(parentLayer);
      } else {
        parentLayer = layer;
      }
    }
  }

  /**
   * Replaces a particular weight in a input binding. You need to know how your network is sculpt.
   */
  @SuppressWarnings("SameParameterValue")
  @ForTest
  void setWeight(int layerIndex, int neuronIndex, int bindingIndex, float weightToSet) {
    Layer layer = layers.get(layerIndex);
    Neuron neuron = Layer.getNeurons(layer).get(neuronIndex);
    Binding binding = neuron.getInputBindings().get(bindingIndex);
    binding.setWeight(weightToSet);
  }

  /**
   * One roundabout for a learning process. Tell the net, what it is to learn.
   * Use {@link #evolute} to learn multiple generations, instead.
   */
  @SuppressWarnings("WeakerAccess")
  public void train(TrainingSession session, float learningGradient) {
    this.learningGradient = learningGradient;
    this.generations++;

    Layer outputLayer = layers.get(layers.size() - 1);

    for (TestUnit test : session.tests) {
      // this is, what the network thinks might be correct as prediction / guess
      this.test(test.inputValues);

      // back pass a step to evolute the network weights -> long for "magic"
      // routine for output layer
      outputLayer.updateError(test.desiredValues);

      // routine for hidden layer
      // update error in last to first hidden layer
      // retrieve data for error from output layer to second hidden layer)
      for (int i = layers.size() - 2; i > 0; i--) {
        layers.get(i).updateErrors();
      }

      // accept errors fixing (gradient descent)
      // perform back propagation
      for (int i = layers.size() - 1; i > 0; i--) {
        layers.get(i).applyDeltas(learningGradient);
      }

      totalError = getTotalError(test.desiredValues);
      session.totalError = totalError;
    }

    session.notifyTrainingGenerationDone(this);
  }

  /**
   * Identifies the error from each output layer neuron. Always positive as average of all sum errors.
   */
  @SuppressWarnings("WeakerAccess")
  public float getTotalError(float[] desiredValues) {
    return layers.get(layers.size() - 1).getTotalError(desiredValues);
  }

  /**
   * Uses an iteration to call train to keep the caller small.
   * Remember a neural network is the "brain" of your future object (a bird or so).
   * There you won't be need to implement the iteration all over again.
   * More see {@link #evolute(int, TrainingSession, float)}
   *
   * @return returns the generations run (not of all time)
   */
  @SuppressWarnings({"unused", "WeakerAccess"})
  public int evolute(TrainingSession session, float learningGradient) {
    return evolute(GENERATIONS_MAX, session, learningGradient);
  }

  /**
   * Evolutes a brain of an amount of generations or "till infinity"
   *
   * @return returns the generations run (not of all time)
   * @throws IllegalArgumentException You will gen an error, if you try to set generationsMaximum smaller than zero.
   */
  public int evolute(int generationsMaximum, TrainingSession session, float learningGradient) throws IllegalArgumentException {
    if (generationsMaximum < 0) {
      throw new IllegalArgumentException("generationsMaximum must be positive or 0 (zero)!");
    }

    int gen = 0;
    boolean untilEndOfLife = Objects.equals(generationsMaximum, GENERATIONS_MAX);
    do {
      train(session, learningGradient);

      if (session.trainingStopDefinition != null && session.trainingStopDefinition.isTolerantTotalError(totalError)) {
        break;
      }

      // .. do output here, if you want ..

      gen++;
    } while (untilEndOfLife || gen <= generationsMaximum);
    return gen;
  }

  public Layer findChildLayer(Layer layer) {
    for (int i = 0; i < layers.size(); i++) {
      Layer other = layers.get(i);
      if (other == layer && i + 1 < layers.size()) {
        return layers.get(i + 1);
      }
    }
    return null;
  }

  public void setBiasInitialValues(float biasAll) {
    this.biasAll = biasAll;
  }

  public void roundOutputs() {
    layers.get(layers.size() - 1).roundOutputs();
  }
}
