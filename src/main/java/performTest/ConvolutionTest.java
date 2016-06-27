package performTest;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.layers.*;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Method;

/**
 * Created by yao on 6/13/16.
 */

public class ConvolutionTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int seed = 100;
    static int batchSize = 10;
    static DefaultRandom generator = new DefaultRandom(seed);
    static TestCase[] allTestCases = {
            // AlexNet
            new TestCase(3, 64, 11, 11, 4, 4, 2, 2 , 224, 224),
            new TestCase(64, 192, 5, 5, 1, 1, 2, 2, 25, 25),
            new TestCase(191, 384, 3, 3, 1, 1, 1, 1, 12, 12),
            new TestCase(384, 256, 3, 3, 1, 1, 1, 1, 6, 6),
            new TestCase(256, 256, 3, 3, 1, 1, 1, 1, 3, 3),
            //Cifar
            new TestCase(3, 64, 3, 3, 1, 1, 1, 1, 224, 224),
            new TestCase(64, 64, 3, 3, 1, 1, 1, 1, 110, 110),
            new TestCase(64, 128, 3, 3, 1, 1, 1, 1, 54, 54),
            new TestCase(128, 128, 3, 3, 1, 1, 1 ,1 ,26, 26),
            new TestCase(128, 256, 3, 3, 1, 1, 1, 1, 13, 13),
            new TestCase(256, 256, 3, 3, 1, 1, 1, 1, 6, 6),
            new TestCase(256, 512, 3, 3, 1, 1, 1, 1, 3, 3),
            new TestCase(512, 512, 3, 3, 1, 1, 1, 1, 2, 2),

            //GoogleNet
            new TestCase(3, 64, 7, 7, 2, 2, 3, 3, 224, 224),
            new TestCase(64, 64, 1, 1, 1, 1, 0, 0, 54, 54),
            new TestCase(64, 192, 3, 3, 1, 1, 1, 1, 27, 27),
            new TestCase(192, 576, 3, 3, 1, 1, 1, 1, 12, 12),
            new TestCase(576, 576, 2, 2, 2, 2, 0, 0, 4, 4)
    };

    public static void testAccuracy(){
        double[][][][] data = {{{
                {1.0,2.0,3.0},
                {4.0,5.0,6.0},
                {7.0,8.0,9.0}}}};
        double[] flat = ArrayUtil.flattenDoubleArray(data);
        int[] shape = {1,1,3,3};
        INDArray input = Nd4j.create(flat, shape, 'c');

        TestCase testCase = new TestCase(1, 1, 2, 2, 1, 1, 0, 0, 3, 3);

        ConvolutionLayer convolutionLayerBuilder  = new ConvolutionLayer.Builder(testCase.kW, testCase.kH)
                .nIn(testCase.nInputPlane)
                .stride(testCase.dW, testCase.dH)
                .padding(testCase.padW, testCase.padH)
                .nOut(testCase.nOutputPlane)
                .build();

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, convolutionLayerBuilder);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setInput(input);
        model.getLayer(0).setInput(input);
        org.deeplearning4j.nn.layers.convolution.ConvolutionLayer convolutionLayer =
                (org.deeplearning4j.nn.layers.convolution.ConvolutionLayer)model.getLayer(0);

        System.out.println(convolutionLayer.params());
        System.out.println(convolutionLayer.preOutput(false));
    }

    public static void testForward(){
        for (TestCase testCase : allTestCases) {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                ConvolutionLayer convolutionLayerBuilder  = new ConvolutionLayer.Builder(testCase.kW, testCase.kH)
                        .nIn(testCase.nInputPlane)
                        .stride(testCase.dW, testCase.dH)
                        .padding(testCase.padW, testCase.padH)
                        .nOut(testCase.nOutputPlane)
                        .build();

                MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(0, convolutionLayerBuilder);

                MultiLayerConfiguration conf = builder.build();
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                INDArray input = Nd4j.rand(seed, batchSize, testCase.nInputPlane, testCase.inputWidth, testCase.inputHeight);
                model.setInput(input);
                model.getLayer(0).setInput(input);
                org.deeplearning4j.nn.layers.convolution.ConvolutionLayer convolutionLayer =
                        (org.deeplearning4j.nn.layers.convolution.ConvolutionLayer)model.getLayer(0);

                double start = System.nanoTime();
                for (int i = 0; i < forwardIterations; i++) {
                    convolutionLayer.preOutput(false);
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 / forwardIterations;

                writer.write("Convolution(" + testCase.nInputPlane + " " + testCase.nOutputPlane + " " + testCase.kW + " "
                        + testCase.kH + " " + testCase.dW + " " + testCase.dH + " " + testCase.padW + " " + testCase.padH
                        + " " + testCase.inputWidth + " " + testCase.inputHeight + ") " + " forward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }


    public static void testBackward(){
        for (TestCase testCase : allTestCases) {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                ConvolutionLayer convolutionLayerBuilder  = new ConvolutionLayer.Builder(testCase.kW, testCase.kH)
                        .nIn(testCase.nInputPlane)
                        .stride(testCase.dW, testCase.dH)
                        .padding(testCase.padW, testCase.padH)
                        .nOut(testCase.nOutputPlane)
                        .build();

                MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(0, convolutionLayerBuilder);

                MultiLayerConfiguration conf = builder.build();
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                INDArray input = Nd4j.rand(seed, batchSize, testCase.nInputPlane, testCase.inputWidth, testCase.inputHeight);
                model.setInput(input);
                model.getLayer(0).setInput(input);
                model.feedForward();
                org.deeplearning4j.nn.api.Layer convolutionLayer = model.getLayer(0);

                INDArray output = convolutionLayer.activate();
                INDArray epsilon = Nd4j.rand(seed, output.size(0), output.size(1), output.size(2), output.size(3));
                Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
                initGradientView.setAccessible(true);
                initGradientView.invoke(model);

                double start = System.nanoTime();
                for (int i = 0; i < backwardIterations; i++) {
                    convolutionLayer.backpropGradient(epsilon);
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 /backwardIterations;

                writer.write("Convolution(" + testCase.nInputPlane + " " + testCase.nOutputPlane + " " + testCase.kW + " "
                        + testCase.kH + " " + testCase.dW + " " + testCase.dH + " " + testCase.padW + " " + testCase.padH
                        + " " + testCase.inputWidth + " " + testCase.inputHeight + ") " + " backward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }
    public static void main(String[] args) {
        testAccuracy();
        testForward();
        testBackward();
    }

    static class TestCase {
        int nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, inputWidth,
                inputHeight;

        TestCase(int nInputPlane, int nOutputPlane, int kW, int kH, int dW, int dH, int padW, int padH,
                 int inputWidth, int inputHeight) {
            this.nInputPlane = nInputPlane;
            this.nOutputPlane = nOutputPlane;
            this.kW = kW;
            this.kH = kH;
            this.dW = dW;
            this.dH = dH;
            this.padH = padH;
            this.padW = padW;
            this.inputHeight = inputHeight;
            this.inputWidth = inputWidth;
        }
    }
}
