package performTest;

import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Method;

/**
 * Created by yansh on 16-6-23.
 */

public class ReLUTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int featureNum = 512;
    static int seed = 100;
    static DefaultRandom generator = new DefaultRandom(seed);

    static INDArray input = Nd4j.rand(featureNum,inputNum);

    static INDArray epsilon = Nd4j.rand(featureNum,featureNum);

    public static void iptestForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0,new RBM.Builder()
                            .activation("relu")
                            .nIn(inputNum)
                            .nOut(featureNum)
                            .build());
            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setInput(input);
            model.getLayer(0).setInput(input);
            model.feedForward();
            INDArray params = model.params();
            model.setParams(params);
            org.deeplearning4j.nn.api.Layer rbm = model.getLayer(0);

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                rbm.activate();
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
            org.deeplearning4j.nn.conf.layers.RBM cnn = new org.deeplearning4j.nn.conf.layers.RBM.Builder()
                    .activation("relu")
                    .nIn(inputNum)
                    .nOut(featureNum)
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
            for (int i = 0; i < forwardIterations; i++) {
                rbm.activate();
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
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0,new RBM.Builder()
                            .activation("relu")
                            .nIn(inputNum)
                            .nOut(featureNum)
                            .build());
            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setInput(input);
            model.getLayer(0).setInput(input);
            INDArray params = model.params();
            model.setParams(params);
            //model.feedForward();

            org.deeplearning4j.nn.api.Layer rbm = model.getLayer(0);

            Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
            initGradientView.setAccessible(true);
            initGradientView.invoke(model);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                rbm.backpropGradient(epsilon);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /backwardIterations;

            writer.write("ReLU(ip) backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    public static void niptestBackward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0,new RBM.Builder()
                            .activation("relu")
                            .nIn(inputNum)
                            .nOut(featureNum)
                            .build());
            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setInput(input);
            model.getLayer(0).setInput(input);

            //model.feedForward();
            org.deeplearning4j.nn.api.Layer rbm = model.getLayer(0);

            Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
            initGradientView.setAccessible(true);
            initGradientView.invoke(model);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                rbm.backpropGradient(epsilon);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /backwardIterations;

            writer.write("ReLU(nip) backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args) {
        niptestForward();
        niptestBackward();
        iptestForward();
        iptestBackward();

    }

}
