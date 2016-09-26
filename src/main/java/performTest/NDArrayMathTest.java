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
    static int sizeLarge = 4096;
    static int sizeMid = 512;
    static int sizeSmall = 32;
    static private INDArray matrixLargeLeft;
    static private INDArray matrixLargeRight;
    static private INDArray matrixLargeVec;
    static private INDArray matrixMidLeft;
    static private INDArray matrixMidRight;
    static private INDArray matrixMidVec;
    static private INDArray matrixSmallLeft;
    static private INDArray matrixSmallRight;
    static private INDArray matrixSmallVec;
    static private int scalar;


    public NDArrayMathTest() {
//        Nd4j.getRandom().setSeed(seed);
        matrixLargeLeft = Nd4j.rand(sizeLarge, sizeLarge);
        if(matrixLargeLeft.data().dataType() == DataBuffer.Type.DOUBLE){
            int a = 1;
        }
        matrixLargeRight = Nd4j.rand(sizeLarge, sizeLarge);
        matrixLargeVec = Nd4j.rand(100L, sizeLarge);
        matrixMidLeft = Nd4j.rand(sizeMid, sizeMid);
        matrixMidRight = Nd4j.rand(sizeMid, sizeMid);
        matrixMidVec = Nd4j.rand(100L, sizeMid);
        matrixSmallLeft = Nd4j.rand(sizeSmall, sizeSmall);
        matrixSmallRight = Nd4j.rand(sizeSmall, sizeSmall);
        matrixSmallVec = Nd4j.rand(100L, sizeSmall);
        scalar = 5;
    }

    public void testMatrixOperation(Operation opt, INDArray left, INDArray right, String printString, int iters) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(new File("dl4jPerformance.csv"), true))) {
            double start = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                switch (opt) {
                    case ADD: left.add(right);
                    case MINUS: left.sub(right);
                    case MULT: left.mmul(right);
                    case DIVID: left.div(right);
                    case ADDMM: left.mmul(right);
                    case ADDMV: left.mmul(right);
                    case POW: Transforms.pow(left,scalar);
                    case LOG: Transforms.log(left);
                    case EXP: Transforms.exp(left);
                    case SQRT: Transforms.sqrt(left);
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
        testMatrixOperation(ADD, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix add operation", 10);
        testMatrixOperation(ADD, matrixMidLeft, matrixMidRight, "512 * 512 matrix add operation", 30);
        testMatrixOperation(ADD, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix add operation", 30);
        testMatrixOperation(MINUS, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix minus operation", 10);
        testMatrixOperation(MINUS, matrixMidLeft, matrixMidRight, "512 * 512 matrix minus operation", 30);
        testMatrixOperation(MINUS, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix minus operation", 30);
        testMatrixOperation(MULT, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix multiply operation", 10);
        testMatrixOperation(MULT, matrixMidLeft, matrixMidRight, "512 * 512 matrix multiply operation", 30);
        testMatrixOperation(MULT, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix multiply operation", 30);
        testMatrixOperation(DIVID, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix divide operation", 10);
        testMatrixOperation(DIVID, matrixMidLeft, matrixMidRight, "512 * 512 matrix divide operation", 30);
        testMatrixOperation(DIVID, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix divide operation", 30);
        testMatrixOperation(ADDMM, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix addmm operation", 10);
        testMatrixOperation(ADDMM, matrixMidLeft, matrixMidRight, "512 * 512 matrix addmm operation", 30);
        testMatrixOperation(ADDMM, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix addmm operation", 30);
        testMatrixOperation(ADDMV, matrixLargeVec, matrixLargeRight, "4096 * 4096 matrix addmv operation", 10);
        testMatrixOperation(ADDMV, matrixMidVec, matrixMidRight, "512 * 512 matrix addmv operation", 30);
        testMatrixOperation(ADDMV, matrixSmallVec, matrixSmallRight, "32 * 32 matrix addmv operation", 30);
        testMatrixOperation(POW, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix add operation", 10);
        testMatrixOperation(POW, matrixMidLeft, matrixMidRight, "512 * 512 matrix add operation", 30);
        testMatrixOperation(POW, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix add operation", 30);
        testMatrixOperation(LOG, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix add operation", 10);
        testMatrixOperation(LOG, matrixMidLeft, matrixMidRight, "512 * 512 matrix add operation", 30);
        testMatrixOperation(LOG, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix add operation", 30);
        testMatrixOperation(EXP, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix add operation", 10);
        testMatrixOperation(EXP, matrixMidLeft, matrixMidRight, "512 * 512 matrix add operation", 30);
        testMatrixOperation(EXP, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix add operation", 30);
        testMatrixOperation(SQRT, matrixLargeLeft, matrixLargeRight, "4096 * 4096 matrix add operation", 10);
        testMatrixOperation(SQRT, matrixMidLeft, matrixMidRight, "512 * 512 matrix add operation", 30);
        testMatrixOperation(SQRT, matrixSmallLeft, matrixSmallRight, "32 * 32 matrix add operation", 30);

    }

    enum Operation{
        ADD, MINUS, MULT, DIVID, ADDMM, ADDMV, POW, LOG, EXP, SQRT
    }
}
