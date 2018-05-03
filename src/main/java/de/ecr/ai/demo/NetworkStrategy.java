package de.ecr.ai.demo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static java.lang.Math.pow;

/**
 * @author Bjoern Frohberg
 */
public class NetworkStrategy {

  private final INeuralNetwork neuralNetwork;

  public NetworkStrategy(INeuralNetwork neuralNetwork) {
    this.neuralNetwork = neuralNetwork;
  }

  public void build(int[] neuronsCount) {
    List<Layer> layers = this.neuralNetwork.getLayers();
    layers.clear();
    Layer parentLayer = null;
    int i = 0;
    for (int neuronCount : neuronsCount) {
      String name = getName(i++, neuronsCount.length);
      Layer layer = new Layer(neuralNetwork, name, neuronCount);
      layers.add(layer);
      if (parentLayer != null) {
        layer.bindTo(parentLayer);
      }
      parentLayer = layer;
    }
  }

  private String getName(int layerIndex, int layers) {
    String name = "Input";
    if (layerIndex > 0) {
      if (layerIndex == layers - 1) {
        name = "Output";
      } else {
        name = "Hidden" + layerIndex;
      }
    }
    return name;
  }

  public float[] propagate(float... inputValues) {
    List<Layer> layers = this.neuralNetwork.getLayers();
    for (int i = 0; i < layers.size(); i++) {
      Layer layer = layers.get(i);
      if (i == 0) {
        layer.setValues(inputValues);
        continue;
      }

      layers.get(i)
        .getNeurons()
        .parallelStream()
        .forEach(n -> n.setOutput(calculateOutput(n)));

      if (i == layers.size() - 1) {
        return layer.getValues();
      }
    }
    return null;
  }

  public void learn(float learningGradient, float... desiredOutput) {

    List<Layer> reversedLayers = new ArrayList<>(this.neuralNetwork.getLayers().stream().skip(1).collect(Collectors.toList()));
    Collections.reverse(reversedLayers);

    // output layer
    reversedLayers.get(0).setDesired(desiredOutput);

    for (Layer layer : reversedLayers) {
      train(layer, learningGradient);
    }

    neuralNetwork.getLayers().forEach(Layer::applyDelta);
  }

  private void train(Layer layer, float learningGradient) {
    List<Neuron> neurons = layer.getNeurons();

    for (Neuron neuron : neurons) {
      float delta = learningGradient * neuron.getDerivative() * neuron.getError();

      for (Binding binding : neuron.getInputBindings()) {
        binding.setDelta(delta * binding.getInput());
      }

      neuron.setBias(neuron.getBias() + delta);
    }
  }

  private float calculateOutput(Neuron neuron) {
    List<Binding> bindings = neuron.getInputBindings();
    float sum = 0f;
    for (Binding binding : bindings) {
      sum += binding.getInput() * binding.getWeight();
    }
    sum += neuron.getBias();

    return neuron.getActivation().activate(sum);
  }

  public double getError() {
    List<Layer> layers = neuralNetwork.getLayers();

    double sum = 0;

    List<Neuron> neurons = layers.get(layers.size() - 1).getNeurons();
    for (Neuron neuron : neurons) {
      sum += pow(neuron.getError(), 2);
    }

    return sum / neurons.size();
  }
}
