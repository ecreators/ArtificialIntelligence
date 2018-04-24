package de.ecr.ai.model;

import de.ecr.ai.model.neuron.activation.IActivationFunction;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

/**
 * @author Bjoern Frohberg
 */
public class NeuralNetworkTest {

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
    float[] values = network.test(0, 1);

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
}