package de.ecr.ai.demo;

/**
 * @author Bjoern Frohberg
 */
public interface IActivationFunction {

  IActivationFunction SIGMOID = new IActivationFunction() {
    @Override
    public float activate(float sum) {
      return (float) (1f / (1 + Math.exp(-sum)));
    }

    @Override
    public float derivate(float output) {
      return output * (1 - output);
    }
  };

  float activate(float sum);

  float derivate(float output);
}
