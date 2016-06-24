package performTest;

import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Method;
/**
 * Created by yansh on 16-6-23.
 */

public class BatchNormalizationTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int featureDim = 512;
    static int seed = 100;
    static DefaultRandom generator = new DefaultRandom(seed);
    //static int bn = new BatchNormalization(FeatureDim);
    static INDArray input = Nd4j.rand(seed, inputNum,featureDim);

    public static void testForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            BatchNormalization batchLayer = new BatchNormalization();

            double[] data = new double[2 * featureDim];
            for (int i = 0; i<featureDim;i++){
                data[i] = i*0.1;
                data[i+featureDim] = i*0.1;
            }
            double[] flat = ArrayUtil.flattenDoubleArray(data);
            int[] shape = {0};
            INDArray grands = Nd4j.create(flat, shape, 'c');

            // TODO: 16-6-24 bias weight
            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, batchLayer);

            MultiLayerConfiguration conf = builder.build();
            NeuralNetConfiguration i1 = conf.getConf(0);
            DefaultParamInitializer paramInit= new DefaultParamInitializer();
            paramInit.init(paramInit.getGradientsFromFlattened(i1,grands),i1,grands);

            org.deeplearning4j.nn.layers.normalization.BatchNormalization batchNormalization
                    = new org.deeplearning4j.nn.layers.normalization.BatchNormalization(i1);

            double start = System.nanoTime();
            for (int i = 0; i < forwardIterations; i++) {
                batchNormalization.preOutput(input);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /forwardIterations;

            writer.write("BatchNormalization forward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void testBackward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            BatchNormalization batchLayer = new BatchNormalization();

            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, batchLayer);

            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setInput(input);
            model.getLayer(0).setInput(input);
            model.feedForward();
            org.deeplearning4j.nn.api.Layer batchNormalization = model.getLayer(0);

            INDArray epsilon = Nd4j.rand(seed, inputNum,featureDim);
            Method initGradientView = model.getClass().getDeclaredMethod("initGradientsView");
            initGradientView.setAccessible(true);
            initGradientView.invoke(model);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                batchNormalization.backpropGradient(epsilon);
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 /backwardIterations;

            writer.write("BatchNormalization backward, " + timeMillis + "\n");
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args){
        testForward();
        testBackward();
    }
}
