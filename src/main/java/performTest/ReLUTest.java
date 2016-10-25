package performTest;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
//import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

import org.deeplearning4j.nn.layers.ActivationLayer;

/**
 * Created by yansh on 16-6-23.
 */

public class ReLUTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int featureNum = 512;
    static int seed = 100;

    static INDArray input = Nd4j.rand(featureNum,inputNum,seed);
    static INDArray epsilon = Nd4j.rand(featureNum,featureNum,seed);

    /*private static ActivationLayer setupipLayer(){
        try{
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(inputNum).nOut(featureNum).activation("relu").build())
                    .layer(1,new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder()
                            .activation("relu")
                            .nIn(inputNum)
                            .nOut(featureNum)
                            .build());
            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setInput(input);
            model.getLayer(1).setInput(input);
            INDArray params = model.params();
            model.setParams(params);
            model.feedForward();
            Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
            initGradientView.setAccessible(true);
            initGradientView.invoke(model);

            ActivationLayer ActivationLayer = (ActivationLayer) model.getLayer(1);
            return ActivationLayer;
        }catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public static void iptestForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            ActivationLayer ActivationLayer = setupipLayer();

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                ActivationLayer.activate(false);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /forwardIterations;

            writer.write("ReLU(ip) forward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void iptestBackward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            ActivationLayer ActivationLayer = setupipLayer();

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                ActivationLayer.backpropGradient(epsilon);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /backwardIterations;

            writer.write("ReLU(ip) backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    private static ActivationLayer setupLayer(){
        try{
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .layer(new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder()
                            .activation("relu")
                            .build())
                    .build();

            int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
            INDArray params = Nd4j.create(1, 51200);
            ActivationLayer activationLayer = LayerFactories.getFactory(conf).create(conf,null,0,params);
            //activationLayer.setBackpropGradientsViewArray(Nd4j.create(1, 0));

            return activationLayer;
        }catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }*/

    private static ActivationLayer setupLayer(){
        try{
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(inputNum).nOut(featureNum).build())
                    .layer(1,new org.deeplearning4j.nn.conf.layers.ActivationLayer.Builder()
                            .activation("relu")
                            .nIn(inputNum)
                            .nOut(featureNum)
                            .build());
            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setInput(input);
            model.getLayer(1).setInput(input);
            model.feedForward();
            Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
            initGradientView.setAccessible(true);
            initGradientView.invoke(model);

            ActivationLayer ActivationLayer = (ActivationLayer) model.getLayer(1);
            return ActivationLayer;
        }catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public static void testForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            ActivationLayer ActivationLayer = setupLayer();

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                ActivationLayer.activate(input);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /forwardIterations;

            writer.write("ReLU(nip) forward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void testBackward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            ActivationLayer ActivationLayer = setupLayer();

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                ActivationLayer.backpropGradient(epsilon);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /backwardIterations;

            writer.write("ReLU(nip) backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args) {
        testForward();
        testBackward();
    }

}
