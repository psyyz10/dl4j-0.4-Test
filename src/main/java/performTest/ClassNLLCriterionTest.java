//package performTest;
//
//import com.sun.org.apache.bcel.internal.generic.FLOAD;
//import org.deeplearning4j.nn.layers.OutputLayer;
//import org.deeplearning4j.nn.layers.factory.LayerFactories;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.lossfunctions.LossFunctions;
//
//import java.io.BufferedWriter;
//import java.io.File;
//import java.io.FileWriter;
//
///**
// * Created by yansh on 16-6-23.
// */
//
//public class ClassNLLCriterionTest {
//
//    static int forwardIterations = 10;
//    static int backwardIterations = 10;
//    static int inputNum = 512;
//    static int featureDim = 512;
//    static int seed = 100;
//    static INDArray input = Nd4j.rand(inputNum, featureDim,seed);
//    static INDArray target = Nd4j.rand(inputNum,inputNum,seed);
//
//    private static OutputLayer setupLayer(){
//        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .layer(new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nIn(inputNum)
//                        .nOut(featureDim)
//                        .build())
//                .build();
//
//        // target.apply1(_ => Generator.uniform(1, 10).asInstanceOf[Float].ceil)
//        target = Nd4j.rand(new int[]{inputNum, featureDim}, Nd4j.getDistributions().createUniform(1.0, 10.0));
//
//        int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
//        INDArray params = Nd4j.create(1, numParams);
//        org.deeplearning4j.nn.layers.OutputLayer outputLayer = LayerFactories.getFactory(conf).create(conf,null,0,params);
//        outputLayer.setBackpropGradientsViewArray(Nd4j.create(1, params.length()));
//        return outputLayer;
//    }
//
//    static public void testForward(){
//        try(BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))){
//            OutputLayer outputLayer = setupLayer();
//            outputLayer.fit(input,target);
//            double start = System.nanoTime();
//            for (int i = 0; i < forwardIterations; i++) {
//                outputLayer.activate(input);
//            }
//            double end = System.nanoTime();
//            double timeMillis = (end - start) / 1e6 /forwardIterations;
//
//            writer.write("ClassNLLCriterion forward, " + timeMillis + "\n");
//        } catch (Exception ex) {
//            ex.printStackTrace();
//        }
//
//    }
//
//    static public void testBackward(){
//        try(BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))){
//            OutputLayer outputLayer = setupLayer();
//            outputLayer.fit(input,target);
//            INDArray output = outputLayer.activate();
//            INDArray epsilon = Nd4j.rand(output.size(0), output.size(1),seed );
//
//            double start = System.nanoTime();
//            for (int i = 0; i < backwardIterations; i++) {
//                outputLayer.backpropGradient(epsilon);
//            }
//            double end = System.nanoTime();
//            double timeMillis = (end - start) / 1e6 /backwardIterations;
//
//            writer.write("ClassNLLCriterion backward, " + timeMillis + "\n");
//        } catch (Exception ex) {
//            ex.printStackTrace();
//        }
//    }
//
//    public static void main(String[] args){
//        testForward();
//        testBackward();
//    }
//}
