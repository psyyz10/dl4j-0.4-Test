import junit.framework.TestCase;
import org.junit.Test;
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
    @Test
    public void testConvolutionTest() {
        ConvolutionTest.testForward();
        ConvolutionTest.testBackward();
    }

    @Test
    public void testMaxPooling() {
        MaxPoolingTest.testForward();
        MaxPoolingTest.testBackward();
    }
}