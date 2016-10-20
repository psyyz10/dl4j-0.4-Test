package performTest;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import static performTest.NDArrayMathTest.Operation.*;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by yao on 9/11/16.
 */
public class NDArrayMathTest {
    static long seed = 100L;
    static int sizeLarge = 40;
    static int sizeMid = 51;
    static int sizeSmall = 32;
    static private INDArray matrixLargeLeft;
    static private INDArray matrixLargeRight;
    static private INDArray matrixLargeResult;
    static private INDArray matrixLargeVec;
    static private INDArray matrixMidLeft;
    static private INDArray matrixMidRight;
    static private INDArray matrixMidResult;
    static private INDArray matrixMidVec;
    static private INDArray matrixSmallLeft;
    static private INDArray matrixSmallRight;
    static private INDArray matrixSmallResult;
    static private INDArray matrixSmallVec;
    static private int scalar;


    private INDArray init(int length, float min, float max, float interval){
        INDArray result = Nd4j.create(length,length);
        float now = min;
        int i=0;
        for (;i< (length*length);i++){
                if (now > max) now = min;
                result.putScalar(i,now);
                now = now+interval;
        }
        return result;
    }

    public NDArrayMathTest() {
//        Nd4j.getRandom().setSeed(seed);
        matrixLargeLeft = init(sizeLarge,-1000,1000,0.5f);
        matrixLargeRight = init(sizeLarge,-1000,1000,0.5f);
        matrixLargeResult = Nd4j.ones(sizeLarge,sizeLarge);
        matrixLargeVec = Nd4j.rand(100L, sizeLarge);
        matrixMidLeft = init(sizeMid,-1000,1000,0.5f);
        matrixMidRight = init(sizeMid,-1000,1000,0.5f);
        matrixMidResult = Nd4j.ones(sizeMid,sizeMid);
        matrixMidVec = Nd4j.rand(100L, sizeMid);
        matrixSmallLeft = init(sizeSmall,-500,500,1);
        matrixSmallRight = init(sizeSmall,-500,500,1);
        matrixSmallResult = Nd4j.ones(sizeSmall,sizeSmall);
        matrixSmallVec = Nd4j.rand(100L, sizeSmall);
        scalar = 5;
    }

    public void testMatrixOperation(Operation opt, INDArray left, INDArray right, INDArray result, String printString, int iters) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            for (int i = 0; i < 10; i++) {
                switch (opt) {
                    case ADD: left.add(right,result);break;
                    case MINUS: left.sub(right,result);break;
                    case MULT: left.mmul(right,result);break;
                    case DIVID: left.div(right,result);break;
                    case ADDMM: left.mmul(right,result);break;
                    case ADDMV: left.mmul(right,result);break;
                    case POW: Transforms.pow(left,scalar);break;
                    case LOG: Transforms.log(left);break;
                    case EXP: Transforms.exp(left);break;
                    case SQRT: Transforms.sqrt(left);break;
                }
            }

            double start = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                switch (opt) {
                    case ADD: left.add(right,result);break;
                    case MINUS: left.sub(right,result);break;
                    case MULT: left.mul(right,result);break;
                    case DIVID: left.div(right,result);break;
                    case ADDMM: left.mmul(right,result);break;
                    case ADDMV: left.mmul(right,result);break;
                    case POW: Transforms.pow(left,scalar);break;
                    case LOG: Transforms.log(left);break;
                    case EXP: Transforms.exp(left);break;
                    case SQRT: Transforms.sqrt(left);break;
                }
            }
            double end = System.nanoTime();
            double timeMillis = (end - start) / 1e6 / iters;
            writer.write(printString + String.format(",%1.3f\n", timeMillis));
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void testMath() {
        testMatrixOperation(POW, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix pow operation", 100);
        testMatrixOperation(POW, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix pow operation", 1000);
        testMatrixOperation(POW, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix pow operation", 100000);
        testMatrixOperation(LOG, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix log operation", 100);
        testMatrixOperation(LOG, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix log operation", 1000);
        testMatrixOperation(LOG, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix log operation", 100000);
        testMatrixOperation(EXP, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix exp operation", 100);
        testMatrixOperation(EXP, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix exp operation", 1000);
        testMatrixOperation(EXP, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix exp operation", 100000);
        testMatrixOperation(SQRT, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix sqrt operation", 100);
        testMatrixOperation(SQRT, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix sqrt operation", 1000);
        testMatrixOperation(SQRT, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix sqrt operation", 100000);
        testMatrixOperation(ADD, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix add operation", 100);
        testMatrixOperation(ADD, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix add operation", 1000);
        testMatrixOperation(ADD, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix add operation", 100000);
        testMatrixOperation(MINUS, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix minus operation", 100);
        testMatrixOperation(MINUS, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix minus operation", 1000);
        testMatrixOperation(MINUS, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix minus operation", 100000);
        testMatrixOperation(MULT, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix multiply operation", 100);
        testMatrixOperation(MULT, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix multiply operation", 1000);
        testMatrixOperation(MULT, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix multiply operation", 100000);
        testMatrixOperation(DIVID, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix divide operation", 100);
        testMatrixOperation(DIVID, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix divide operation", 1000);
        testMatrixOperation(DIVID, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix divide operation", 100000);
        testMatrixOperation(ADDMM, matrixLargeLeft, matrixLargeRight, matrixLargeResult, "4096 * 4096 matrix addmm operation", 100);
        testMatrixOperation(ADDMM, matrixMidLeft, matrixMidRight, matrixMidResult, "512 * 512 matrix addmm operation", 1000);
        testMatrixOperation(ADDMM, matrixSmallLeft, matrixSmallRight, matrixSmallResult, "32 * 32 matrix addmm operation", 100000);
        testMatrixOperation(ADDMV, matrixLargeVec, matrixLargeRight, matrixLargeVec, "4096 * 4096 matrix addmv operation", 100);
        testMatrixOperation(ADDMV, matrixMidVec, matrixMidRight, matrixMidVec, "512 * 512 matrix addmv operation", 1000);
        testMatrixOperation(ADDMV, matrixSmallVec, matrixSmallRight, matrixSmallVec, "32 * 32 matrix addmv operation", 100000);
    }

    enum Operation{
        ADD, MINUS, MULT, DIVID, ADDMM, ADDMV, POW, LOG, EXP, SQRT
    }
}
