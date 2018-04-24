package de.ecr.ai.model;

import de.ecr.ai.model.annotation.LearningData;
import de.ecr.ai.model.neuron.IBindableSourceNeuron;
import de.ecr.ai.model.neuron.Neuron;

import java.security.SecureRandom;
import java.util.Random;

import static java.text.MessageFormat.format;

/**
 * Binding to a parent neuron (neuron in previous layer [hidden layer or input layer]
 *
 * @author Bjoern Frohberg
 */
public final class Binding {

  private static final Random RANDOM = new SecureRandom();

  private final IBindableSourceNeuron parentNeuron;
  private final String childName;
  private float weight;

  /**
   * Creates a new instance an force weight to be a random floating number between -1 and 1.
   */
  public Binding(Neuron child, IBindableSourceNeuron parentNeuron) {
    this.parentNeuron = parentNeuron;
    this.childName = child.getName();
    randomizeWeight();
    if (parentNeuron == null) {
      throw new IllegalArgumentException("Your neuron may not be null! " + parentNeuron);
    }
  }

  /**
   * For visualization
   */
  public String getName() {
    return format("{0}:-( {1} )->:{2}", parentNeuron.getName(), weight, childName);
  }

  /**
   * Initial weight between -1 and 1
   */
  public void randomizeWeight() {
    weight = RANDOM.nextFloat() * 2 - 1;
  }

  /**
   * Use it during loading a neural network
   */
  public void setWeight(float weight) {
    this.weight = weight;
  }

  /**
   * Returns a calculated balanced value depending on the parent neuron output value
   */
  public float calculateOutput() {
    return weight * parentNeuron.getOutputValue();
  }

  /**
   * Learning rule. Update algorithm for weight.<br/>
   * This is the dynamic a neural network does.<br/>
   * <b>Learning for a neural network is a nearing way to get closer to an expected value.</b><br/>
   * Every error a network calculates from a predicted output will<br/>
   * change every weight on previous bindings in each layer, but balanced by<br/>
   * output weights on output bindings.
   */
  public void learn(float learningGradient, float error) {
    float in = parentNeuron.getOutputValue();
    weight += learningGradient * in * error;
  }

  /**
   * Returns the weight for a connected input value (output of parent neuron).<br/>
   * It balances the input value to possibly be able to predict a close valid output value.<br/>
   * The combination of (input x weight = output) represents everything, what a network does,<br/>
   * When it does not learn.
   */
  @LearningData
  public float getWeight() {
    return weight;
  }
}
