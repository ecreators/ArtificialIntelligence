package de.ecr.ai.model;

/**
 * Represents a delegate with parameter
 *
 * @author Bjoern Frohberg
 */
public interface Action<T> {

  void invoke(T args);
}
