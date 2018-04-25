package de.ecr.ai.model.test;

/**
 * Defines a stop condition, when your network is valid enough for you
 *
 * @author Bjoern Frohberg
 */
public interface ITrainingStop {

  ITrainingStop STOP_AT_5_PERCENT_ERRORS = t -> t <= 5 / 100f;

  /**
   * Return {@code true}, if you accept the remaining errors through all test units
   */
  boolean isTolerantTotalError(float totalError);
}
