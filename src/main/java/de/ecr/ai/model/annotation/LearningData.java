package de.ecr.ai.model.annotation;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Inherited;
import java.lang.annotation.Target;

/**
 * For understanding: this value will be changed by the network during training
 *
 * @author Bjoern Frohberg
 */
@Documented
@Inherited
@Target(ElementType.METHOD)
public @interface LearningData {
}
