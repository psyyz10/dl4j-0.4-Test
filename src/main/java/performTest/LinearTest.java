package performTest;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.RBM;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Method;

/**
 * Created by yansh on 16-6-23.
 */

public class LinearTest{
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int seed = 100;
    static DefaultRandom generator = new DefaultRandom(seed);

    static TestCase[] allTestCases = {
            new TestCase(256 * 6 * 6, 4096),//, 3000, 3900),
            new TestCase(4096, 4096),//1500, 2000),
            new TestCase(256 * 5 * 5, 128),// 65, 90),
            new TestCase(512, 512),// 21, 30),
            new TestCase(512, 10),// 2, 2),
            new TestCase(28 * 4 * 4, 768)//, 28, 40)
    };

    public static void testForward() {
        for (TestCase testCase : allTestCases) {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                INDArray input = Nd4j.rand(inputNum,testCase.inputSize,seed);

                MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(0,new RBM.Builder()
                                .nIn(testCase.inputSize)
                                .nOut(testCase.outputSize)
                                .visibleUnit(RBM.VisibleUnit.LINEAR)
                                .build());
                MultiLayerConfiguration conf = builder.build();
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setInput(input);
                model.getLayer(0).setInput(input);
                model.feedForward();

                org.deeplearning4j.nn.api.Layer rbm = model.getLayer(0);

                double start = System.nanoTime();
                for (int i = 0; i < forwardIterations; i++) {
                    rbm.activate();
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 /forwardIterations;

                writer.write("Linear(" + testCase.inputSize + " " + testCase.outputSize  + ") forward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void testBackward(){
        for (TestCase testCase : allTestCases) {
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                INDArray input = Nd4j.rand(inputNum,testCase.inputSize,seed);

                MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                        .list()
                        .layer(0,new RBM.Builder()
                                .nIn(testCase.inputSize)
                                .nOut(testCase.outputSize)
                                .visibleUnit(RBM.VisibleUnit.LINEAR)
                                .build());
                MultiLayerConfiguration conf = builder.build();
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setInput(input);
                model.getLayer(0).setInput(input);
                model.feedForward();
                org.deeplearning4j.nn.api.Layer rbm = model.getLayer(0);

                INDArray output = rbm.activate();
                INDArray epsilon = Nd4j.rand(output.size(0), output.size(1), seed);
                Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
                initGradientView.setAccessible(true);
                initGradientView.invoke(model);

                double start = System.nanoTime();
                for (int i = 0; i < backwardIterations; i++) {
                    rbm.backpropGradient(epsilon);
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 /backwardIterations;

                writer.write("Linear(" + testCase.inputSize + " " + testCase.outputSize  + ") backward, " + timeMillis + "\n");
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
        int inputSize,outputSize;

        TestCase(int iS, int oS) {
            this.inputSize = iS;
            this.outputSize = oS;
        }
    }
}