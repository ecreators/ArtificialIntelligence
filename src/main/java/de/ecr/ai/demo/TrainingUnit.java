package de.ecr.ai.demo;

/**
 * @author Bjoern Frohberg
 */
public class TrainingUnit {

  private float[] testInputs;
  private float[] expectedOutpus;

  public float[] getTestValues() {
    return testInputs;
  }

  public float[] getExpectationValues() {
    return expectedOutpus;
  }

  public TrainingUnit setInputs(float... inputs) {
    this.testInputs = inputs;
    return this;
  }

  public TrainingUnit setDesiredOutputs(float... expectedOutputs) {
    this.expectedOutpus = expectedOutputs;
    return this;
  }
}
