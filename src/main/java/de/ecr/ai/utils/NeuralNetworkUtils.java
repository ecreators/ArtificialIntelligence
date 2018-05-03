package de.ecr.ai.utils;

/**
 * A helper class for calculations
 *
 * @author Bjoern Frohberg
 */
public final class NeuralNetworkUtils {

  private NeuralNetworkUtils() {
  }

  /**
   * Calculates a proper number of hidden layers neurons for this particular amount of inputs and hidden layer count.
   * This was a tricky part to identify the best neural network count for hidden neurons
   */
  public static int calculateHiddenNeuronCount(int hiddenLayers, int inputs) {
    return (int) squareSum(Math.pow(2, inputs) * hiddenLayers);
  }

  /**
   * Calculate the square sum to a number (e.g. 3 => 3 + 2 + 1 = 6)
   */
  public static double squareSum(double number) {
    return number > 1 ? number + squareSum(number - 1) : number;
  }
}
