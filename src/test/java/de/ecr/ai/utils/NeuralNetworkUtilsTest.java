package de.ecr.ai.utils;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

/**
 * @author Bjoern Frohberg
 */
public class NeuralNetworkUtilsTest {

  @Test
  public void testSquareSum() {
    assertThat(NeuralNetworkUtils.squareSum(3), is(equalTo(3d + 2 + 1)));
    assertThat(NeuralNetworkUtils.squareSum(5), is(equalTo(5d + 4 + 3 + 2 + 1)));
  }
}