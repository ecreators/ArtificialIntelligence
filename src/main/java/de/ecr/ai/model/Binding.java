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
          throw new IllegalArgumentException("Your neuron may not be null! parentNeuron");
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
     * Returns the weight for a connected input value (output of parent neuron).<br/>
     * It balances the input value to possibly be able to predict a close valid output value.<br/>
     * The combination of (input x weight = output) represents everything, what a network does,<br/>
     * When it does not learn.
     */
    @LearningData
    public float getWeight() {
        return weight;
    }

    /**
     * Identifies {@code true}, if parentNeuron instance is equal in reference
     */
    public boolean isParentNeuron(Neuron parentNeuron) {
        return this.parentNeuron == parentNeuron;
    }

  /**
   * Update weight for child neuron, based on it error
   */
  public void updateWeight(float learningGradient, Neuron childNeuron) {
    weight += calculateWeightDelta(childNeuron, parentNeuron.getOutputValue(), learningGradient);
  }

  private static float calculateWeightDelta(Neuron childNeuron, float outputValue, float learningGradient) {
    return learningGradient * outputValue * childNeuron.getError();
  }
}
