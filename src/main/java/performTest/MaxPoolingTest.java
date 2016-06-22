package performTest;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

//import org.deeplearning4j.nn.layers.convolution.subsampling;


/**
 * Created by yao on 6/7/16.
 */
public class MaxPoolingTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int batchSize = 100;
    static int nInputPlane = 3;
    static int height = 100;
    static int weight = 512;
    static int seed = 100;
    static DefaultRandom generator = new DefaultRandom(seed);
    static MaxPoolingTest.TestCase[] allTestCases = {
            new TestCase(3, 3, 2, 2), // AlexNet, GoogleNet
            new TestCase(2, 2, 2, 2), // Cifar, CifarLocal
            new TestCase(3, 3, 1, 1), //GoogleNet
            new TestCase(3, 3, 3, 3) // MNIST
    };

    static INDArray input = Nd4j.rand(seed, batchSize, nInputPlane, height, weight);

    public static void testForward(){
        for (TestCase testCase : allTestCases) {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                SubsamplingLayer subSamplingLayer  = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(testCase.kW, testCase.kH)
                        .stride(testCase.dW, testCase.dH)
                        .build();

                MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(0, subSamplingLayer);

                MultiLayerConfiguration conf = builder.build();
                NeuralNetConfiguration i1 = conf.getConf(0);

                org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer subsamplingLayer =
                        new org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer(i1, input);


                double start = System.nanoTime();
                for (int i = 0; i < forwardIterations; i++) {
                    subsamplingLayer.activate();
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 /forwardIterations;

                writer.write("MaxPooling(" + testCase.kW + " " + testCase.kH + " " + testCase.dW +
                        " " + testCase.dH + ") " + " forward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }


    public static void testBackward(){
        for (TestCase testCase : allTestCases) {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                SubsamplingLayer subSamplingLayer  = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(testCase.kW, testCase.kH)
                        .stride(testCase.dW, testCase.dH)
                        .build();

                MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(0, subSamplingLayer);

                MultiLayerConfiguration conf = builder.build();
                NeuralNetConfiguration i1 = conf.getConf(0);

                org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer subsamplingLayer =
                        new org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer(i1, input);

                INDArray output = subsamplingLayer.activate();
                INDArray epsilon = Nd4j.rand(seed, output.size(0), output.size(1), output.size(2), output.size(3));

                double start = System.nanoTime();
                for (int i = 0; i < backwardIterations; i++) {
                    subsamplingLayer.backpropGradient(epsilon);
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 / backwardIterations;

                writer.write("MaxPooling(" + testCase.kW + " " + testCase.kH + " " + testCase.dW +
                    " " + testCase.dH + ") " + " backward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }
    public static void main(String[] args){
        testForward();
        testBackward();
    }

    static class TestCase {
        int kW, kH, dW, dH;

        TestCase(int kW, int kH, int dW, int dH) {
            this.kW = kW;
            this.kH = kH;
            this.dW = dW;
            this.dH = dH;
       }
    }
}
