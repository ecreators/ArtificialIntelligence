package de.ecr.ai.demo;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Bjoern Frohberg
 */
class Layer {

  private final List<Neuron> neurons = new ArrayList<>();
  private final INeuralNetwork network;
  private final String name;

  Layer(INeuralNetwork network, String name, int neurons) {
    this.network = network;
    this.name = name;
    for (int i = 0; i < neurons; i++) {
      newNeuron(this);
    }
  }

  private static final String arrayOpen = "[ ";
  private static final String arrayClose = " ]";
  private static final String arrayDelimiter = ", ";

  public void printDebugInfo() {
    String name = getName();
    int neurons = countNeurons();
    Float[] biasWeights = this.neurons.stream().map(Neuron::getBias).toArray(Float[]::new);
    Float[][] neuronWeightsList = this.neurons.stream().map(n -> n.getInputBindings().stream().map(Binding::getWeight).toArray(Float[]::new)).toArray(Float[][]::new);

    System.out.println("{");
    System.out.println("  name: " + name);
    System.out.println("  neurons: " + neurons);
    System.out.println("  weights: " + join(neuronWeightsList, arrayOpen, arrayClose, arrayDelimiter, arrayOpen, arrayClose, arrayDelimiter));
    System.out.println("  biases: " + join(biasWeights, arrayOpen, arrayClose, arrayDelimiter));
    System.out.println("}");
  }

  private static String join(Float[][] neuronWeightsList, String prefix, String suffix, String delimiter, String unitPrefix, String unitSuffix, String unitDelimiter) {
    StringBuilder sb = new StringBuilder(prefix);
    for (Float[] neuronWeights : neuronWeightsList) {
      if (sb.length() > prefix.length()) {
        sb.append(delimiter);
      }

      sb.append(join(neuronWeights, unitPrefix, unitSuffix, unitDelimiter));
    }
    sb.append(suffix);
    return sb.toString();
  }

  private static String join(Float[] neuronWeights, String prefix, String suffix, String delimiter) {
    StringBuilder sbb = new StringBuilder(prefix);
    for (Float weight : neuronWeights) {
      if (sbb.length() > prefix.length()) {
        sbb.append(delimiter);
      }
      sbb.append(weight);
    }
    sbb.append(suffix);
    return sbb.toString();
  }

  private void newNeuron(Layer layer) {
    Neuron neuron = new Neuron(layer);
    neurons.add(neuron);
  }

  void bindTo(Layer parentLayer) {
    for (Neuron child : neurons) {
      for (Neuron parent : parentLayer.neurons) {
        Binding binding = new Binding(parent);
        parent.getOutputBindings().add(binding);
        child.getInputBindings().add(binding);
      }
    }
  }

  List<Neuron> getNeurons() {
    return neurons;
  }

  public float[] getValues() {
    float[] values = new float[neurons.size()];
    for (int i = 0; i < neurons.size(); i++) {
      values[i] = neurons.get(i).getOutput();
    }
    return values;
  }

  public void setValues(float[] values) {
    for (int i = 0; i < neurons.size(); i++) {
      Neuron neuron = neurons.get(i);
      neuron.setOutput(values[i]);
    }
  }

  public void setDesired(float[] desiredOutput) {
    for (int i = 0; i < neurons.size(); i++) {
      Neuron neuron = neurons.get(i);
      neuron.setDesired(desiredOutput[i]);
    }
  }

  public boolean isOutputLayer() {
    return network.isOutputLayer(this);
  }

  public void applyDelta() {
    neurons.forEach(Neuron::applyDelta);
  }

  public float getWeights() {
    return (float) neurons.stream().mapToDouble(Neuron::getWeights).sum();
  }

  public int countNeurons() {
    return neurons.size();
  }

  public String getName() {
    return name;
  }
}