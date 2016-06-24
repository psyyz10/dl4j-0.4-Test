package performTest;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Field;

/**
 * Created by yansh on 16-6-23.
 */
public class DropoutTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int InputNum = 1000;
    static int FeatureDim = 512;
    static int seed = 100;
    static DefaultRandom generator = new DefaultRandom(seed);
    static double[] allTestCases = {0.3,0.4,0.5};

    public static INDArray input = Nd4j.rand(InputNum,FeatureDim);

    public static void testForward(){
        for(double testCase : allTestCases){
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dropOut(0.5)
                        .list()
                        .layer(0, new OutputLayer.Builder().build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
                dropoutMaskField.setAccessible(true);

                NeuralNetConfiguration i1 = conf.getConf(0);

                org.deeplearning4j.nn.layers.OutputLayer outputLayer =
                        new org.deeplearning4j.nn.layers.OutputLayer(i1, input);

                double start = System.nanoTime();
                for (int i = 0; i < forwardIterations; i++) {
                    outputLayer.activate();
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 /forwardIterations;

                writer.write("Dropout(" + testCase + ") " + " forward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void testBackward(){
        for(double testCase : allTestCases){
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dropOut(0.5)
                        .list()
                        .layer(0, new OutputLayer.Builder().build())
                        .build();

                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
                dropoutMaskField.setAccessible(true);

                NeuralNetConfiguration i1 = conf.getConf(0);

                org.deeplearning4j.nn.layers.OutputLayer outputLayer =
                        new org.deeplearning4j.nn.layers.OutputLayer(i1, input);

                INDArray output = outputLayer.activate();
                INDArray epsilon = Nd4j.rand(seed, output.size(0), output.size(1));

                double start = System.nanoTime();
                for (int i = 0; i < backwardIterations; i++) {
                    outputLayer.backpropGradient(epsilon);
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 /backwardIterations;

                writer.write("Dropout(" + testCase + ") " + " backward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }
}
