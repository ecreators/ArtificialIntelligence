package de.ecr.ai.model;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.core.Is.is;

/**
 * @author Bjoern Frohberg
 */
public class NeuralNetworkTest {

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
}