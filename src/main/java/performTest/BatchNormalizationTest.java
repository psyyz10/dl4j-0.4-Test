package performTest;

import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import org.deeplearning4j.nn.api.Layer;


/**
 * Created by yansh on 16-6-23.
 */

public class BatchNormalizationTest {
    static int forwardIterations = 10;
    static int backwardIterations = 10;
    static int inputNum = 100;
    static int featureDim = 512;
    static int seed = 100;
    static INDArray input = Nd4j.rand(inputNum,featureDim,seed);

    protected static Layer setupActivations(int nIn, int nOut){
        BatchNormalization bN = new BatchNormalization.Builder().nIn(nIn).nOut(nOut).build();
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layer(bN).build();

        Map<String, INDArray> paramTable = new HashMap<>();
        double[] data = new double[2 * featureDim];
        for (int i = 0; i<featureDim;i++){
            data[i] = i*0.1;
            data[i+featureDim] = i*0.1;
        }
        INDArray Data = Nd4j.create(data);
        BatchNormalizationParamInitializer paramsInit = new BatchNormalizationParamInitializer();
        paramsInit.init(paramTable,conf,Data);

        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer =  LayerFactories.getFactory(conf).create(conf, null, 0, params);
        return layer;

    }

    public static void testForward(){
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            Layer batchNormalization = setupActivations(inputNum,featureDim);
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
            Layer layer = setupActivations(inputNum,featureDim);
            INDArray epsilon = Nd4j.rand(inputNum,featureDim);
            layer.setBackpropGradientsViewArray(Nd4j.create(1,2*featureDim));
            layer.preOutput(input);
            layer.backpropGradient(epsilon);

            double start = System.nanoTime();
            for (int i = 0; i < backwardIterations; i++) {
                layer.backpropGradient(epsilon);
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
