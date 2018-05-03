package de.ecr.ai.demo;

import java.security.SecureRandom;
import java.util.Random;

/**
 * @author Bjoern Frohberg
 */
class Binding {

  private static final Random RANDOM = new SecureRandom();

  private final Neuron parent;
  private float weight;
  private float delta;
  private float deltaError;

  Binding(Neuron parent) {
    this.parent = parent;
    this.weight = RANDOM.nextFloat() - RANDOM.nextFloat();
  }

  public float getInput() {
    return parent.getOutput();
  }

  public void setWeight(float weight) {
    this.weight = weight;
  }

  public float getWeight() {
    return weight;
  }

  public void setDelta(float delta) {
    this.delta = delta;
    this.deltaError = delta * weight;
  }

  public float getDeltaError() {
    return deltaError;
  }

  void applyDelta() {
    weight += delta;
  }
}
