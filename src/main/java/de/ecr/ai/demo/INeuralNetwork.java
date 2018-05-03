package de.ecr.ai.demo;

import java.util.List;

/**
 * @author Bjoern Frohberg
 */
public interface INeuralNetwork {

  List<Layer> getLayers();

  boolean isOutputLayer(Layer layer);

  boolean isInputLayer(Layer layer);
}