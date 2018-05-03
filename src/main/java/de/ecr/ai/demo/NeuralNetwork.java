package de.ecr.ai.demo;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Bjoern Frohberg
 */
public class NeuralNetwork implements INeuralNetwork {

  private int generation;

  public void print() {
    layers.forEach(Layer::printDebugInfo);
  }

  public void invokeTraining(float learnGradient, List<TrainingUnit> units) {
    units.forEach(unit -> invokeTrainingUnit(learnGradient, unit));
    this.generation++;
  }

  public int getGeneration() {
    return generation;
  }

  public void invokeTrainingUnit(float learnGradient, TrainingUnit unit) {
    test(unit.getTestValues());
    learn(learnGradient, unit.getExpectationValues());
  }

  public float getWeights() {
    return (float) layers.stream().mapToDouble(Layer::getWeights).sum();
  }

  public double getError() {
    return strategy.getError();
  }

  private final NetworkStrategy strategy;
  private final List<Layer> layers;

  public NeuralNetwork() {
    this.layers = new ArrayList<>();
    this.strategy = new NetworkStrategy(this);
  }

  public void build(int[] neuronsCount) {
    strategy.build(neuronsCount);
  }

  public void build(NetworkRack rack) {
    build(rack.getData());
  }

  public float[] test(float... inputs) {
    return strategy.propagate(inputs);
  }

  public void learn(float learnGradient, float... desiredValues) {
    strategy.learn(learnGradient, desiredValues);
  }

  public static int fact(double value) {
    return fact((int) Math.round(value));
  }

  public static int fact(int value) {
    return value > 1 ? value + fact(value - 1) : value;
  }

  @Override
  public List<Layer> getLayers() {
    return this.layers;
  }

  @Override
  public boolean isOutputLayer(Layer layer) {
    return layers.indexOf(layer) == layers.size() - 1;
  }

  @Override
  public boolean isInputLayer(Layer layer) {
    return layers.indexOf(layer) == 0;
  }
}