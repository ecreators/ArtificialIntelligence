package de.ecr.ai.model.annotation;

import java.lang.annotation.*;

/**
 * Determine a method only for use in a test
 *
 * @author Bjoern Frohberg
 */
@Target(ElementType.METHOD)
@Inherited
@Documented
@Retention(RetentionPolicy.RUNTIME)
public @interface ForTest {
}
