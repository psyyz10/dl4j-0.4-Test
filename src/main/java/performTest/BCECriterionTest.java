package performTest;

import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Method;

/**
 * Created by yansh on 16-6-23.
 */
public class BCECriterionTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int featureDim = 512;
    static int seed = 100;

    static INDArray input = Nd4j.rand(featureDim,inputNum);
    static INDArray target = Nd4j.rand(featureDim,inputNum);

    private static OutputLayer setupLayer(){
        org.deeplearning4j.nn.conf.layers.OutputLayer bce =
                new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .nIn(inputNum).nOut(featureDim)
                        .build();
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .layer(bce)
                .build();

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        org.deeplearning4j.nn.layers.OutputLayer outputLayer = LayerFactories.getFactory(conf).create(conf,null,0,params);
        outputLayer.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));

        return outputLayer;
    }

    private static OutputLayer setupMulLayer(){
        try{
            org.deeplearning4j.nn.conf.layers.OutputLayer bce =
                    new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                            .nIn(inputNum).nOut(featureDim)
                            .build();
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, bce);
            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setInput(input);
            model.getLayer(0).setInput(input);
            model.feedForward();
            Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
            initGradientView.setAccessible(true);
            initGradientView.invoke(model);

            OutputLayer outputLayer = (OutputLayer) model.getLayer(0);

            INDArray output = outputLayer.activate();
            outputLayer.fit(input,output);

            return outputLayer;
        }catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public static void testForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            OutputLayer outputLayer = setupLayer();
// TODO: 16-6-29 target
            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                outputLayer.preOutput(target);
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
            OutputLayer outputLayer = setupLayer();
            INDArray output = outputLayer.activate(input);
            outputLayer.fit(input,output);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                outputLayer.backpropGradient(target);
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
