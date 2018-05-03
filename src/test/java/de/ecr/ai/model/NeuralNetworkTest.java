package de.ecr.ai.model;

import de.ecr.ai.demo.ArrayUtils;
import de.ecr.ai.demo.NetworkRack;
import de.ecr.ai.demo.TrainingUnit;
import de.ecr.ai.model.neuron.activation.IActivationFunction;
import de.ecr.ai.model.test.TestUnit;
import de.ecr.ai.model.test.TrainingSession;
import de.ecr.ai.utils.NeuralNetworkUtils;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static de.ecr.ai.demo.NeuralNetwork.fact;
import static java.lang.Math.pow;
import static java.lang.Math.round;
import static java.text.MessageFormat.format;
import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;
import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

/**
 * @author Bjoern Frohberg
 */
public class NeuralNetworkTest {

  private int inputs = 2;

  private NetworkRack rack = new de.ecr.ai.demo.NetworkRack()
    .setInputNeurons(inputs)
    .addHiddenLayer(fact(pow(2, inputs)))
    .setOutputNeurons(1);

  @Test
  public void xorTest() {
    // given
    de.ecr.ai.demo.NeuralNetwork network = new de.ecr.ai.demo.NeuralNetwork();
    network.build(rack);

    // when
    float learnGradient = 0.65f;
    float errorTolerance = 0.1f;
    TrainingUnit test00 = new TrainingUnit().setInputs(0, 0).setDesiredOutputs(0);
    TrainingUnit test01 = new TrainingUnit().setInputs(0, 1).setDesiredOutputs(1);
    TrainingUnit test10 = new TrainingUnit().setInputs(1, 0).setDesiredOutputs(1);
    TrainingUnit test11 = new TrainingUnit().setInputs(1, 1).setDesiredOutputs(0);
    List<TrainingUnit> units = Arrays.asList(test00, test01, test10, test11);

    do {
      network.invokeTraining(learnGradient, units);
      System.out.println(format("Generation: {0} -> Error: {1} ~ Weights: {2}", network.getGeneration(), network.getError(), network.getWeights()));
    } while (network.getError() >= errorTolerance);

    // then
    network.print();

    int[] results = ArrayUtils.round(network.test(test01.getTestValues()));
    assertThat(round(results[0]), is(equalTo(ArrayUtils.round(test01.getExpectationValues())[0])));

    results = ArrayUtils.round(network.test(test10.getTestValues()));
    assertThat(round(results[0]), is(equalTo(ArrayUtils.round(test10.getExpectationValues())[0])));

    results = ArrayUtils.round(network.test(test00.getTestValues()));
    assertThat(round(results[0]), is(equalTo(ArrayUtils.round(test00.getExpectationValues())[0])));

    results = ArrayUtils.round(network.test(test11.getTestValues()));
    assertThat(round(results[0]), is(equalTo(ArrayUtils.round(test11.getExpectationValues())[0])));
  }

  @Test
  public void testSumFact() {
    int fact = fact(4);
    assertThat(fact, is(equalTo(10)));
  }

  /**
   * Test, how to setup a network manually with "build"-method and usage of "readMemory"
   * <p>
   * Tests
   * - creating neutwork
   * - create layers
   * - create neurons
   * - create bindings to neurons on an upper layer (parent layer) as fully meshed
   * - read settings into MemoryData-Object
   */
  @Test
  public void buildANeuralNetworkSetting() {
    // given
    int inputs = 2;
    int hiddenLayersCount = 1;
    int hiddens = 2;
    int outputs = 1;
    boolean softMax = false;

    // when
    NeuralNetwork network = new NeuralNetwork();
    network.build(inputs, hiddenLayersCount, hiddens, outputs, softMax);

    // then
    // no exception please!

    MemoryData memory = network.readMemory();
    assertThat(memory, is(not(nullValue())));
    assertThat(memory.inputs, is(equalTo(inputs)));
    assertThat(memory.outputs, is(equalTo(outputs)));
    assertThat(memory.hiddenLayers, is(equalTo(hiddenLayersCount)));
    assertThat(memory.hiddenNeurons, is(equalTo(hiddens)));
  }

  /**
   * Tests, how to push / propagate input values through the network and return the output layer neuron values as array
   * <p>
   * - setup with "build"
   * - pass input values into input layer neurons "as is" one-by-one
   * - loop through all layers and call propate on each layer and neuron - calculate neuron output, using inputbindings
   * (input value times weight) and pass through activation function, until and including output layer
   * - return output layer neuron output values as array of floating numbers between 0 and 1
   * <p>
   * because the initial weights are random, this test cannot assert for specific output values
   */
  @Test
  public void propagateValues() {
    // given
    int inputs = 2;
    int hiddenLayersCount = 1;
    int hiddens = 2;
    int outputs = 1;
    boolean softMax = false;
    NeuralNetwork network = new NeuralNetwork();
    network.build(inputs, hiddenLayersCount, hiddens, outputs, softMax);

    // when
    Float[] values = network.test(0, 1);

    // then
    assertThat(values, is(notNullValue()));
    assertThat(values.length, is(equalTo(outputs)));
    assertThat(values.length, is(not(equalTo(0f))));
  }

  /**
   * Tests, how to use json and MemoryData as build setup for the network.
   * <p>
   * MemoryData is the reminding from the latest training configuration
   */
  @Test
  public void testMemoryData() {
    // given
    int inputs = 2;
    int hiddenLayersCount = 1;
    int hiddens = 2;
    int outputs = 1;
    boolean softMax = false;

    MemoryData data = new MemoryData();
    data.inputs = inputs;
    data.outputs = outputs;
    data.hiddenLayers = hiddenLayersCount;
    data.hiddenNeurons = hiddens;
    data.softMaxUsed = softMax;
    data.generations = 58800;

    // when
    NeuralNetwork network = new NeuralNetwork();
    String json = network.saveToJson(data);
    network.loadMemoryDataFromJson(json);

    // then
    MemoryData memory = network.readMemory();
    assertThat(memory, is(not(nullValue())));
    assertThat(memory.inputs, is(equalTo(data.inputs)));
    assertThat(memory.outputs, is(equalTo(data.outputs)));
    assertThat(memory.hiddenLayers, is(equalTo(data.hiddenLayers)));
    assertThat(memory.hiddenNeurons, is(equalTo(data.hiddenNeurons)));
    assertThat(memory.generations, is(equalTo(data.generations)));
  }

  @Test
  public void testSimpleBackpropagation() {
    // given
    NeuralNetwork network = new NeuralNetwork();
    network.build(1, 1, 1, 1, false);

    // when
    final float rawInputValue = 2;
    final float boundaryInputValue = 2f;

    float hiddenWeight = 0.5f;
    float hiddenBias = 0f;
    network.setWeight(1, 0, 0, hiddenWeight);

    float outputWeight = 0.5f;
    float outputBias = 0f;
    network.setWeight(2, 0, 0, outputWeight);

    float normalizedValue = rawInputValue / boundaryInputValue;
    float output = network.test(normalizedValue)[0];

    // then
    // hidden neuron
    float sum = rawInputValue / boundaryInputValue * hiddenWeight + hiddenBias;
    IActivationFunction hiddenActivation = IActivationFunction.SIGMOID;
    float activated = hiddenActivation.activate(sum);

    // output neuron
    sum = activated * outputWeight + outputBias;
    IActivationFunction outputActivation = IActivationFunction.SIGMOID;
    activated = outputActivation.activate(sum);

    assertThat(output, is(equalTo(activated)));
  }

  /**
   * A generation is the point from input values over propagation to gradient descent error correction
   * and weight adjustment
   */
  @Test
  public void testOneGenerationXor() {
    // given
    NeuralNetwork brain = new NeuralNetwork();
    int inputs = 2;
    int hiddenLayersCount = 1;
    int hiddenNeurons = NeuralNetworkUtils.calculateHiddenNeuronCount(hiddenLayersCount, inputs);
    brain.build(inputs, hiddenLayersCount, hiddenNeurons, 1, false);

    // when
    TrainingSession session = createXorSmokeTestSession();

    // then

    float old = session.totalError;
    brain.train(session, 0.35f);
    // is equal as train only
    // brain.evolute(1, session, 0.35f);

    assertThat(session.totalError, is(not(equalTo(old))));
  }

  /**
   * Train multiple times of generations to get a valid prediction
   */
  @Test
  @Ignore // see fixme - this is tricky
  public void doSmokeTest() {

    // FIXME -> the training works, but it need some time to identify why the training resolves zero values
    // after rounding. Seems not the network did not already identified the correct weighting.
    // need some debugging.

    // given
    NeuralNetwork brain = new NeuralNetwork();
    brain.setBiasInitialValues(1);

    int inputs = 2;
    int hiddenLayersCount = 1;
    int hiddenNeurons = NeuralNetworkUtils.calculateHiddenNeuronCount(hiddenLayersCount, inputs);
    brain.build(inputs, hiddenLayersCount, hiddenNeurons, 1, false);
//    brain.roundOutputs();

    // when
    TrainingSession session = createXorSmokeTestSession();

    // no training
    int[] expectedSmokeTestResults = {0, 1, 1, 0};
    // demo, no training - for debugging
    int[] testResults = readSmokeTestResults(brain, session);

    // with training
    int generations = 1000;
    float learningGradient = 0.35f;

    // when
    for (int gen = 0; gen < generations; gen++) {
      brain.train(session, learningGradient);
      System.out.println(String.format("total error, gen# %d: %.4f", gen, session.totalError));
    }

    // then
    testResults = new int[]{
      rInt(brain.test(0, 0)[0]), // proper desired XOR 0
      rInt(brain.test(0, 1)[0]), // proper desired XOR 1
      rInt(brain.test(1, 0)[0]), // proper desired XOR 1
      rInt(brain.test(1, 1)[0]) // proper desired XOR 0
    };
    assertThat(testResults, is(equalTo(expectedSmokeTestResults)));
  }

  private static int[] readSmokeTestResults(NeuralNetwork brain, TrainingSession session) {
    // test does never train! - only reading
    float test00 = brain.test(session.tests.get(0).inputValues)[0]; // having always 1 output, so index 0
    float test01 = brain.test(session.tests.get(1).inputValues)[0];
    float test10 = brain.test(session.tests.get(2).inputValues)[0];
    float test11 = brain.test(session.tests.get(3).inputValues)[0];
    return new int[]{
      rInt(test00),
      rInt(test01),
      rInt(test10),
      rInt(test11)};
  }

  private static int rInt(float value) {
    return round(value);
  }

  private TrainingSession createXorSmokeTestSession() {
    TrainingSession session = new TrainingSession();
    // this is the xor smoke test
    // Description: the neural network shall be able to identify, when inputs identify "is xor"
    // XOR: true, if only one of 2 inputs is 1
    session.tests.add(newTestUnit(asList(0f, 0f), singletonList(0f)));
    session.tests.add(newTestUnit(asList(0f, 1f), singletonList(1f)));
    session.tests.add(newTestUnit(asList(1f, 0f), singletonList(1f)));
    session.tests.add(newTestUnit(asList(1f, 1f), singletonList(0f)));
    return session;
  }


  /**
   * Retrieve a test unit given only by inputs and desired output values
   */
  private static TestUnit newTestUnit(List<Float> inputs, List<Float> desiredOutputs) {
    TestUnit testUnit = new TestUnit();
    testUnit.inputValues = toFloatArray(inputs);
    testUnit.desiredValues = toFloatArray(desiredOutputs);
    return testUnit;
  }

  @Test
  public void testToArray() {
    // given
    List<Float> arr = new ArrayList<>();
    arr.add(1f);
    arr.add(2f);
    arr.add(3f);

    // when
    float[] floatings = toFloatArray(arr);

    // then
    assertThat("No equal array match!", Arrays.equals(floatings, new float[]{1, 2, 3}), is(true));
  }

  /**
   * Converts List&lt;Float&gt; to float[]
   */
  private static float[] toFloatArray(List<Float> floatings) {
    float[] floats = new float[floatings.size()];
    AtomicInteger i = new AtomicInteger();
    floatings.forEach(f -> floats[i.getAndIncrement()] = f);
    return floats;
  }

}