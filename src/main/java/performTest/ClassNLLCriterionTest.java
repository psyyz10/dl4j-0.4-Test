package performTest;

import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.layers.OutputLayer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Created by yansh on 16-6-23.
 */
public class ClassNLLCriterionTest {

    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 512;
    static int featureDim = 512;
    static int seed = 100;
    static DefaultRandom generator = new DefaultRandom(seed);
    static INDArray input = Nd4j.rand(seed, inputNum, featureDim);
    static INDArray target = Nd4j.rand(seed, inputNum);

    static public void testForward(){
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))){
            org.deeplearning4j.nn.conf.layers.RBM nll = new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                    .nIn(inputNum).nOut(inputNum)
                    .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .build();
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .layer(nll)
                    .build();

            int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
            INDArray params = Nd4j.create(1, numParams);
            org.deeplearning4j.nn.layers.feedforward.rbm.RBM rbm = LayerFactories.getFactory(conf).create(conf,null,0,params);
            rbm.fit(input);

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                rbm.activate(target);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /forwardIterations;

            writer.write("ClassNLLCriterion forward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    static public void testBackward(){
        try(BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))){
            org.deeplearning4j.nn.conf.layers.RBM nll = new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                    .nIn(inputNum).nOut(inputNum)
                    .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .build();
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .layer(nll)
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

            writer.write("ClassNLLCriterion backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args){
        testForward();
        testBackward();
    }
}
