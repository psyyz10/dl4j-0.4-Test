package performTest;

/*import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.*;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.layers.OutputLayer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;*/

/**
 * Created by yansh on 16-6-23.
 */
public class DropoutTest {
   /* static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int InputNum = 100;
    static int FeatureDim = 512;
    static int seed = 100;
    static double[] allTestCases = {0.3,0.4,0.5};

    public static INDArray input = Nd4j.rand(FeatureDim,InputNum,seed);

    private static OutputLayer setupLayer(){
        try{
            NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                    .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                            .dropOut(0.5)
                            .nIn(InputNum)
                            .nOut(FeatureDim)
                            .build())
                    .build();
            Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
            dropoutMaskField.setAccessible(true);

            int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
            INDArray params = Nd4j.create(1, numParams);
            org.deeplearning4j.nn.layers.OutputLayer outputLayer = LayerFactories.getFactory(conf).create(conf,null,0,params);
            outputLayer.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));

            return outputLayer;

        }catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public static OutputLayer setupMulLayer(){
        try{
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .dropOut(0.5)
                .list()
                .layer(0, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
                        .nIn(InputNum)
                        .nOut(FeatureDim)
                        .build())
                .build();

        Field dropoutMaskField = BaseLayer.class.getDeclaredField("dropoutMask");
        dropoutMaskField.setAccessible(true);

        NeuralNetConfiguration i1 = conf.getConf(0);
        int numParams = LayerFactories.getFactory(i1).initializer().numParams(i1,true);
        INDArray params = Nd4j.create(1, numParams);
        OutputLayer outputLayer = LayerFactories.getFactory(i1).create(i1,null,0,params);

        return outputLayer;
        }catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public static void testForward(){
        for(double testCase : allTestCases){
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
                OutputLayer outputLayer = setupLayer();

                double start = System.nanoTime();
                for (int i = 0; i < forwardIterations; i++) {
                    outputLayer.preOutput(input);
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
                OutputLayer outputLayer = setupLayer();

                INDArray output = outputLayer.activate(input);
                INDArray epsilon = Nd4j.rand(output.size(0), output.size(1),seed);
                outputLayer.fit(input,output);

                double start = System.nanoTime();
                for (int i = 0; i < backwardIterations; i++) {
                    outputLayer.backpropGradient(epsilon);
                }
                double end = System.nanoTime();
                double timeMillis = (end - start) / 1e6 / backwardIterations;

                writer.write("Dropout(" + testCase + ") " + " backward, " + timeMillis + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void main(String[] args){
        testForward();
        testBackward();
    }*/
}
