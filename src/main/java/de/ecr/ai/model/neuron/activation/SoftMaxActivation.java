package de.ecr.ai.model.neuron.activation;

import de.ecr.ai.model.Layer;
import de.ecr.ai.model.neuron.Neuron;
import de.ecr.ai.model.neuron.OutputNeuron;

import java.util.List;

/**
 * Use for a classificational result.
 * This is a result with multiple outputs to get only one output near by or exact 1 and every output else 0.
 *
 * @author Bjoern Frohberg
 */
public class SoftMaxActivation implements IActivationFunction {

  private final Neuron neuron;

  public SoftMaxActivation(OutputNeuron n) {
    this.neuron = n;
  }

  @Override
  public float activate(float sum) {
    // calculates a softmax a single result with 1 for all possibilities (classification)
    List<Neuron> neurons = Layer.getNeurons(neuron.getLayer());
    float total = (float) neurons.stream().mapToDouble(n -> Math.exp(n.getOutputValue())).sum();
    return (float) (Math.exp(sum) / total);
  }
}
