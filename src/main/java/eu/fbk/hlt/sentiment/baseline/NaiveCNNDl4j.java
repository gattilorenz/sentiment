package eu.fbk.hlt.sentiment.baseline;

import com.google.inject.*;
import com.google.inject.name.Named;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import eu.fbk.hlt.data.LabeledSentences;
import eu.fbk.hlt.data.WordVectors;
import eu.fbk.hlt.sentiment.AbstractModel;
import eu.fbk.hlt.sentiment.util.DatasetProvider;
import eu.fbk.hlt.sentiment.util.SentimentParameters;
import eu.fbk.hlt.sentiment.util.Stopwatch;
import org.apache.commons.cli.ParseException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A naive model that utilizes a single Convolution layer using deeplearning4j library
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class NaiveCNNDl4j extends AbstractModel {
    MultiLayerNetwork net = null;

    public static final int DEFAULT_SENTENCE_SIZE = 50;
    public static final int DEFAULT_ITERATION_PER_BATCH = 20;
    public static final int DEFAULT_SEED = 241;
    public static final int DEFAULT_WINDOW_SIZE = 2;
    public static final int DEFAULT_DIM_MULTIPLIER = 2;

    protected int seed = DEFAULT_SEED;
    protected int windowSize = DEFAULT_WINDOW_SIZE;
    protected int iterationsPerBatch = DEFAULT_ITERATION_PER_BATCH;
    protected int sentenceSize = DEFAULT_SENTENCE_SIZE;
    protected int lookupDim = embeddings.getDim() * DEFAULT_DIM_MULTIPLIER;

    @Inject
    public NaiveCNNDl4j(@Named("classes") List<String> classes, WordVectors embeddings, AnnotationPipeline pipeline) throws Exception {
        super(classes, embeddings, pipeline);
    }

    private void buildNeuralNet() throws Exception {
        int numRows = 1;
        int numColumns = sentenceSize;
        int numChannels = embeddings.getDim();
        int iterations = iterationsPerBatch;
        int[] kernel = new int[] {1, windowSize};
        int[] stride = new int[] {1, 1};

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(3)
            .layer(0, new ConvolutionLayer.Builder(kernel, stride)
                .nIn(numChannels*numColumns)
                .nOut(lookupDim)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, kernel, stride)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(classes.size())
                .weightInit(WeightInit.XAVIER)
                .activation("softmax")
                .build())
            .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder, numRows, numColumns, numChannels);

        net = new MultiLayerNetwork(builder.build());
        net.init();
        net.setListeners(Collections.singletonList(new ScoreIterationListener(2)));
    }

    private List<DataSet> currentBatch = new ArrayList<>();

    /**
     * Convert an arbitrary sentence to a fixed-sized representation
     *
     * @param input
     * @return
     */
    private INDArray alignInput(INDArray input) {
        INDArray alignedInput = Nd4j.zeros(sentenceSize*input.columns());
        int limit = input.rows() > sentenceSize ? sentenceSize : input.rows();
        for (int i = 0; i < limit; i++) {
            alignedInput.put(new INDArrayIndex[]{NDArrayIndex.interval(i*input.columns(),(i+1)*input.columns())}, input.getRow(i));
        }
        return alignedInput;
    }

    @Override
    protected void train(INDArray input, INDArray label) {
        if (net == null) {
            try {
                buildNeuralNet();
            } catch (Exception e) {
                logger.warn("Can't init neural network");
                return;
            }
        }

        currentBatch.add(new DataSet(alignInput(input), label));

        if (currentBatch.size() % DEFAULT_BATCH_SIZE == 0) {
            Stopwatch watch = Stopwatch.start();
            net.fit(DataSet.merge(currentBatch));
            logger.info("A batch has been trained. "+((double)watch.click()/1000)+"s");
            currentBatch.clear();
        }
    }

    @Override
    public INDArray predict(INDArray input) {
        if (net == null) {
            try {
                buildNeuralNet();
            } catch (Exception e) {
                logger.warn("Can't init neural network");
            }
        }

        return net.output(alignInput(input));
    }

    public int getSeed() {
        return seed;
    }

    public NaiveCNNDl4j setSeed(int seed) throws Exception {
        buildNeuralNet();
        this.seed = seed;
        return this;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public NaiveCNNDl4j setWindowSize(int windowSize) throws Exception {
        buildNeuralNet();
        this.windowSize = windowSize;
        return this;
    }

    public int getIterationsPerBatch() {
        return iterationsPerBatch;
    }

    public NaiveCNNDl4j setIterationsPerBatch(int iterationsPerBatch) throws Exception {
        buildNeuralNet();
        this.iterationsPerBatch = iterationsPerBatch;
        return this;
    }

    public int getSentenceSize() {
        return sentenceSize;
    }

    public NaiveCNNDl4j setSentenceSize(int sentenceSize) throws Exception {
        buildNeuralNet();
        this.sentenceSize = sentenceSize;
        return this;
    }

    /**
     * Here we automatically bootstrap
     *  our environment with just a little bit
     *  of a custom WordVectors construction
     *  via the DatasetProvider class
     */
    public static void main(String[] args) throws ParseException, Nd4jBackend.NoAvailableBackendException {
        SentimentParameters params = new SentimentParameters(args);
        Injector injector = Guice.createInjector(new DatasetProvider(params));
        NaiveCNNDl4j project = injector.getInstance(NaiveCNNDl4j.class);
        project.train(injector.getInstance(LabeledSentences.class));
        if (params.interactiveMode) {
            project.interactive();
        }
    }
}
