package performTest;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.layers.RBM;


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

    static INDArray input = Nd4j.rand(seed, inputNum, featureDim);
    static INDArray target = Nd4j.rand(seed,inputNum);

    public static void testForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, new RBM.Builder()
                            .activation("sigmoid")
                            .lossFunction(LossFunctions.LossFunction.XENT).build());

            MultiLayerConfiguration conf = builder.build();
            NeuralNetConfiguration i1 = conf.getConf(0);

            org.deeplearning4j.nn.layers.feedforward.rbm.RBM subsamplingLayer =
                    new org.deeplearning4j.nn.layers.feedforward.rbm.RBM(i1, input);

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                subsamplingLayer.activate(target);
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
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, new RBM.Builder()
                            .activation("sigmoid")
                            .lossFunction(LossFunctions.LossFunction.XENT).build());

            MultiLayerConfiguration conf = builder.build();
            NeuralNetConfiguration i1 = conf.getConf(0);

            org.deeplearning4j.nn.layers.feedforward.rbm.RBM RBMLayer =
                    new org.deeplearning4j.nn.layers.feedforward.rbm.RBM(i1, input);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                RBMLayer.derivativeActivation(target);
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
