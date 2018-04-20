package de.ecr.ai.model;

import java.util.List;

/**
 * Describes a neural network configuration. The BRAIN!
 *
 * @author Bjoern Frohberg
 */
public class MemoryData {
	
	public int                     inputs;
	public int                     hiddenLayers;
	public int                     hiddenNeurons;
	public int                     outputs;
	public boolean                 softMaxUsed;
	public List<List<List<Float>>> weights;
	public List<List<Float>>       biases;
	public String                  networkName;
	public long                    generations;
	public float                   learningGradient;
}
