import junit.framework.TestCase;
import performTest.ClassNLLCriterionTest;
import performTest.ConvolutionTest;
import performTest.MaxPoolingTest;

/**
 * Created by yao on 6/21/16.
 */

public class DL4JTest extends TestCase {
//    public void testClassNLLCriterion(){
//        ClassNLLCriterionTest.testForward();
//        ClassNLLCriterionTest.testBackward();
//    }

    public void testConvolutionTest() {
        ConvolutionTest.testForward();
        ConvolutionTest.testBackward();
    }

    public void testMaxPooling() {
        MaxPoolingTest.testForward();
        MaxPoolingTest.testBackward();
    }
}