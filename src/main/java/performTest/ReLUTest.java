package performTest;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Created by yansh on 16-6-23.
 */

public class ReLUTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int featureNum = 512;
    static int seed = 100;
    //static int noutputPanel = 500;
    static DefaultRandom generator = new DefaultRandom(seed);

    static INDArray input = Nd4j.rand(inputNum,featureNum);

    static INDArray epsilon = Nd4j.rand(inputNum,featureNum);

    public static void iptestForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            DenseLayer ipModel = new DenseLayer.Builder().activation("relu")
                    //.nOut(noutputPanel)
                    .build();

            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0,ipModel);

            MultiLayerConfiguration conf = builder.build();
            NeuralNetConfiguration i1 = conf.getConf(0);

            org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer denseLayer =
                    new org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer(i1, input);

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                denseLayer.activate();
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /forwardIterations;

            writer.write("ReLU(ip) forward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void niptestForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            // TODO: 16-6-23
            DenseLayer nipModel = new DenseLayer.Builder().activation("relu")
                    //.nOut(noutputPanel)
                    .build();

            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0,nipModel);

            MultiLayerConfiguration conf = builder.build();
            NeuralNetConfiguration i1 = conf.getConf(0);

            org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer denseLayer =
                    new org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer(i1, input);

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                denseLayer.activate();
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /forwardIterations;

            writer.write("ReLU(nip) forward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void iptestBackward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            DenseLayer ipModel = new DenseLayer.Builder().activation("relu")
                    //.nOut(noutputPanel)
                    .build();

            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0,ipModel);

            MultiLayerConfiguration conf = builder.build();
            NeuralNetConfiguration i1 = conf.getConf(0);

            org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer denseLayer =
                    new org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer(i1, input);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                denseLayer.backpropGradient(epsilon);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /backwardIterations;

            writer.write("ReLU(ip) backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    public static void niptestBackward(){

    }

    public static void main(String[] args) {
        iptestForward();
        iptestBackward();
        niptestForward();
        niptestBackward();
    }
}
