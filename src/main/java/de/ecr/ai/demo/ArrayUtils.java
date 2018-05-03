package de.ecr.ai.demo;

/**
 * @author Bjoern Frohberg
 */
public final class ArrayUtils {

  private ArrayUtils() { }

  public static int[] round(float[] values) {
    int[] outputs = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      outputs[i] = Math.round(values[i]);
    }
    return outputs;
  }
}
