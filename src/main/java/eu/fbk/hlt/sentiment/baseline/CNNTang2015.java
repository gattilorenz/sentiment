package eu.fbk.hlt.sentiment.baseline;

import com.google.inject.*;
import com.google.inject.name.Named;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import eu.fbk.hlt.data.LabeledSentences;
import eu.fbk.hlt.data.WordVectors;
import eu.fbk.hlt.sentiment.AbstractModel;
import eu.fbk.hlt.sentiment.nn.ConvolutionLayer;
import eu.fbk.hlt.sentiment.nn.Pipeline;
import eu.fbk.hlt.sentiment.nn.duyu.*;
import eu.fbk.hlt.sentiment.util.DatasetProvider;
import eu.fbk.hlt.sentiment.util.SentimentParameters;
import org.apache.commons.cli.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * The definition of a Convolutional Neural Network for the sentence representation by Duyu Tang
 * "Learning Semantic Representations of Users and Products for Document Level Sentiment Classification" (from ACL2015)
 *
 * Three filters of size 1,2 and 3
 *  Each filter is a lookup layer, linear, average, tanh
 * Average filter
 * Softmax at the top
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class CNNTang2015 extends AbstractModel {
    final static Logger logger = LoggerFactory.getLogger(CNNTang2015.class);

    public static final double RESULT_THESHOLD = 0.001;

    protected LabeledSentences dataset;
    protected ArrayList<Pipeline> net;
    protected Pipeline softmax;

    @Inject
    public CNNTang2015(@Named("classes") List<String> classes, WordVectors embeddings, AnnotationPipeline pipeline) throws Exception {
        super(classes, embeddings, pipeline);
        buildNeuralNet(embeddings.getDim(), embeddings.getDim());
    }

    private void buildNeuralNet(int wordDim, int lookupDim) throws Exception {
        this.net = new ArrayList<>();
        //Create a set of layers for each of the window sizes
        Pipeline connect = new Pipeline(new MultiConnectLayer(new int[]{lookupDim, lookupDim, lookupDim}));
        LinearLayer linear = new LinearLayer(lookupDim, 5);
        double rndBase = -0.01;
        linear.randomize(new Random(), -1.0 * rndBase, rndBase);
        softmax = connect
            .after(new AverageLayer(lookupDim*3, lookupDim))
            .after(linear)
            .after(new SoftmaxLayer(5));

        for (int windowSize = 1; windowSize <= 3; windowSize++) {
            Pipeline net = new Pipeline(new ConvolutionLayer(windowSize, wordDim, lookupDim));
            net.link(connect, windowSize-1);
            this.net.add(net);
        }
    }

    private int counter = 0;
    private double lossV = 0.0;

    @Override
    protected void train(INDArray input, INDArray label) {
        SoftmaxLayer layer = (SoftmaxLayer) softmax.getInputLayer();

        setInput(input);
        for (Pipeline conv : net) {
            conv.forward();
        }

        int trueLabel = getMaxIndex(label);
        lossV += -Math.log(layer.output[trueLabel]);
        for (int i = 0; i < layer.outputG.length; i++) {
            layer.outputG[i] = 0.0;
        }

        if (layer.output[trueLabel] < RESULT_THESHOLD) {
            layer.outputG[trueLabel] = 1.0 / RESULT_THESHOLD;
        } else {
            layer.outputG[trueLabel] = 1.0 / layer.output[trueLabel];
        }

        for (Pipeline conv : net) {
            conv.backward();
        }

        for (Pipeline conv : net) {
            conv.update(0.03);
        }

        for (Pipeline conv : net) {
            conv.clearGrad();
        }

        if (++counter % DEFAULT_BATCH_SIZE == 0) {
            logger.info("lossV/lossC = "+lossV+"/"+(lossV/counter));
        }
    }

    private int getMaxIndex(INDArray array) {
        int maxInd = 0;
        double max = array.getDouble(maxInd);

        for (int i = 1; i < array.columns(); i++) {
            if (array.getDouble(i) > max) {
                maxInd = i;
                max = array.getDouble(maxInd);
            }
        }

        return maxInd;
    }

    @Override
    protected INDArray predict(INDArray input) {
        setInput(input);
        for (Pipeline conv : net) {
            conv.forward();
        }
        return Nd4j.create(((SoftmaxLayer) softmax.getInputLayer()).output);
    }

    private void setInput(INDArray input) {
        assert input.rows() > 0;
        assert input.columns() > 1;

        int dim = input.columns();
        int size = input.rows() < 3 ? 3 : input.rows();
        double[] rawInput = new double[dim*size];
        for (int i = 0; i < input.rows(); i++) {
            for (int j = 0; j < dim; j++) {
                rawInput[j+i*dim] = input.getRow(i).getDouble(j);
            }
        }
        for (int i = input.rows(); i  < size; i++) {
            Arrays.fill(rawInput, i*dim, (i+1)*dim, 0.0);
        }
        for (Pipeline conv : net) {
            ConvolutionLayer layer = (ConvolutionLayer) conv.getInputLayer();
            layer.setInput(rawInput);
        }
    }

    /**
     * Here we automatically bootstrap
     *  our environment with just a little bit
     *  of a custom WordVectors construction
     *  via the DatasetProvider class
     */
    public static void main(String[] args) throws ParseException {
        SentimentParameters params = new SentimentParameters(args);
        Injector injector = Guice.createInjector(new DatasetProvider(params));
        CNNTang2015 project = injector.getInstance(CNNTang2015.class);
        project.train(injector.getInstance(LabeledSentences.class));
    }
}
