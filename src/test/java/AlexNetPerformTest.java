import junit.framework.TestCase;
import modelPerfoemTest.AlexNetTest;
import org.junit.Test;

/**
 * Created by yansh on 16-9-18.
 */
public class AlexNetPerformTest extends TestCase {
    @Test
    public void testAlex() {
        //AlexNetTest.testForward();
        AlexNetTest.testBackward();
    }
}
