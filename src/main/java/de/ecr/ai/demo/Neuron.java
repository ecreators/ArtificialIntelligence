package de.ecr.ai.demo;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Bjoern Frohberg
 */
class Neuron {

  private final Layer layer;
  private final List<Binding> outputBindings;
  private final List<Binding> inputBindings;
  private IActivationFunction activator = IActivationFunction.SIGMOID;

  private float bias;
  private float output;
  private float derivative;
  private float error;

  public Neuron(Layer layer) {
    this.layer = layer;
    this.outputBindings = new ArrayList<>();
    this.inputBindings = new ArrayList<>();
    this.bias = 1;
  }

  List<Binding> getInputBindings() {
    return inputBindings;
  }

  List<Binding> getOutputBindings() {
    return outputBindings;
  }

  public float getOutput() {
    return output;
  }

  public void setBias(float bias) {
    this.bias = bias;
  }

  public float getBias() {
    return bias;
  }

  IActivationFunction getActivation() {
    return activator;
  }

  void setOutput(float value) {
    this.output = value;
    this.derivative = activator.derivate(this.output);
  }

  void setDesired(float desired) {
    this.error = (desired - this.output);
  }

  public float getError() {

    // output neuron

    if (layer.isOutputLayer()) {
      return error;
    }

    // hidden neuron: error given by bindings comming from next layer to this neuron

    float sumError = 0;
    for (Binding binding : outputBindings) {
      sumError += binding.getDeltaError();
    }
    return sumError;
  }

  float getDerivative() {
    return derivative;
  }

  public void applyDelta() {
    inputBindings.forEach(Binding::applyDelta);
  }

  public float getWeights() {
    return (float)inputBindings.stream().mapToDouble(Binding::getWeight).sum() + bias;
  }
}
