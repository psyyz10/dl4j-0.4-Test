package performTest;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Created by yansh on 16-6-23.
 */
public class BCECriterionTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int featureDim = 512;
    static int seed = 100;

    static DefaultRandom generator = new DefaultRandom(seed);

    static INDArray input = Nd4j.rand(featureDim,inputNum);
    static INDArray target = Nd4j.rand(featureDim,inputNum);

    public static void testForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            org.deeplearning4j.nn.conf.layers.RBM cnn = new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                    .activation("sigmoid")
                    .nIn(inputNum)
                    .nOut(featureDim)
                    .lossFunction(LossFunctions.LossFunction.XENT)
                    .build();

            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .layer(cnn)
                    .build();

            int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
            INDArray params = Nd4j.create(1, numParams);
            org.deeplearning4j.nn.layers.feedforward.rbm.RBM rbm = LayerFactories.getFactory(conf).create(conf,null,0,params);
            rbm.fit(input);

            // TODO: 16-6-27 target apply

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                rbm.preOutput(target);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /forwardIterations;
             writer.write("BCECriterion forward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void testBackward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            org.deeplearning4j.nn.conf.layers.RBM cnn = new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                    .activation("sigmoid")
                    .nIn(inputNum).nOut(inputNum)
                    .lossFunction(LossFunctions.LossFunction.XENT)
                    .build();

            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .layer(cnn)
                    .build();

            int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
            INDArray params = Nd4j.create(1, numParams);
            org.deeplearning4j.nn.layers.feedforward.rbm.RBM rbm = LayerFactories.getFactory(conf).create(conf,null,0,params);
            rbm.fit(input);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                rbm.derivativeActivation(target);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /backwardIterations;

            writer.write("BCECriterion backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args){
        testForward();
        testBackward();
    }

}
