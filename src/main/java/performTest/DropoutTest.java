package performTest;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

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

    public static INDArray input = Nd4j.rand(FeatureDim,InputNum,seed);

    public static void testForward(){
        for(double testCase : allTestCases){
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .dropOut(0.5)
                        .list()
                        .layer(0, new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                                .nIn(InputNum)
                                .nOut(FeatureDim)
                                .build())
                        .build();

                Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
                dropoutMaskField.setAccessible(true);

                NeuralNetConfiguration i1 = conf.getConf(0);

                int numParams = LayerFactories.getFactory(i1).initializer().numParams(i1,true);
                INDArray params = Nd4j.create(1, numParams);
                org.deeplearning4j.nn.layers.feedforward.rbm.RBM rbm = LayerFactories.getFactory(i1).create(i1,null,0,params);
                rbm.fit(input);

                double start = System.nanoTime();
                for (int i = 0; i < forwardIterations; i++) {
                    rbm.activate();
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
                        .layer(0, new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                                .nIn(InputNum)
                                .nOut(FeatureDim)
                                .build())
                        .build();

                Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
                dropoutMaskField.setAccessible(true);

                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setInput(input);
                model.getLayer(0).setInput(input);
                model.feedForward();
                org.deeplearning4j.nn.api.Layer rbm = model.getLayer(0);
                INDArray output = rbm.activate();
                INDArray epsilon = Nd4j.rand(output.size(0), output.size(1),seed);

                Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
                initGradientView.setAccessible(true);
                initGradientView.invoke(model);

                double start = System.nanoTime();
                for (int i = 0; i < backwardIterations; i++) {
                    rbm.backpropGradient(epsilon);
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 /backwardIterations;

                writer.write("Dropout(" + testCase + ") " + " backward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void main(String[] args){
        testForward();
        testBackward();
    }
}
