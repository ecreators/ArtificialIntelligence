package de.ecr.ai.model.test;

import de.ecr.ai.model.Action;
import de.ecr.ai.model.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * A bunch of {@link TestUnit}s, a valid generation run-through collected in this class.
 *
 * @author Bjoern Frohberg
 */
public final class TrainingSession {

    /**
     * In here are tests to train the {@link NeuralNetwork}
     */
    public final List<TestUnit> tests = new ArrayList<>();

    /**
     * You will need to know, how good your network is. This values represents how wrong<br/>
     * your network actual is.<br/>
     * As low this value is, your network is the best. This value is a percentage as<br/>
     * factor with a value 0 (best) to 1 (terrible wrong).
     * Here is a strategy for this value:
     * <ul>
     * <li>Your network won't get best only after decades of testing on a complex network
     * with a lot of inputs.<br/></li>
     * <li>
     * Your network don't need to be perfect. If you will use it to improve it itself,<br/>
     * then it never will be perfect, but you don't need to be it.<br/>
     * </li>
     * <li>
     * Your network will do its work, even then if it has as nearing value to zero. For<br/>
     * example a value of 0.2 or 0.05 is a very good result for your network.
     * </li>
     * <li>
     * But for first, you should train your network as much generations (TestSessions)
     * until your total error value becomes into this range.
     * </li>
     * </ul>
     */
    public float totalError;

    /**
     * You can define a training stop by a total error. Your network will (if set) stop training, if your total
     * error was passed. After this threshold your network only will learn 1 generation, instead of many.
     */
    public ITrainingStop trainingStopDefinition;

    @SuppressWarnings("WeakerAccess")
    public Action<NeuralNetwork> onGenerationDone;

    /**
     * Invoke training end
     */
    public void notifyTrainingGenerationDone(NeuralNetwork neuralNetwork) {
        Action<NeuralNetwork> handler = this.onGenerationDone;
        if (handler != null) {
            handler.invoke(neuralNetwork);
        }
    }
}
