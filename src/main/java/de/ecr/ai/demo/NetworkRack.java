package de.ecr.ai.demo;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Bjoern Frohberg
 */
public class NetworkRack {

  private int inputNeurons;
  private final List<Integer> hiddenNeurons;
  private int outputNeurons;

  public NetworkRack() {
    hiddenNeurons = new ArrayList<>();
  }

  public NetworkRack setInputNeurons(int inputs) {
    this.inputNeurons = inputs;
    return this;
  }

  public NetworkRack setOutputNeurons(int outputs) {
    this.outputNeurons = outputs;
    return this;
  }

  public NetworkRack addHiddenLayer(int hiddens) {
    this.hiddenNeurons.add(hiddens);
    return this;
  }

  public int[] getData() {
    int[] data = new int[2 + hiddenNeurons.size()];
    data[0] = inputNeurons;
    data[data.length - 1] = outputNeurons;
    for (int i = 1; i < 1 + hiddenNeurons.size(); i++) {
      data[i] = hiddenNeurons.get(i - 1);
    }
    return data;
  }
}
