package eu.fbk.hlt.sentiment.baseline;

import com.google.inject.AbstractModule;
import com.google.inject.Guice;
import com.google.inject.Injector;
import com.google.inject.Provides;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.pipeline.BinarizerAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import eu.fbk.hlt.data.Dataset;
import eu.fbk.hlt.data.DatasetRepository;
import eu.fbk.hlt.data.LabeledSentences;
import eu.fbk.hlt.data.WordVectors;
import eu.fbk.hlt.sentiment.util.CLIOptionBuilder;
import eu.fbk.hlt.sentiment.util.DatasetProvider;
import eu.fbk.hlt.sentiment.util.SentimentParameters;
import eu.fbk.hlt.sentiment.util.Stopwatch;
import org.apache.commons.cli.*;
import org.deeplearning4j.eval.Evaluation;
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
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.UiServer;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.ejml.simple.SimpleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import java.io.*;
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
 * @deprecated
 */
public class CNNTang2015dl4j {
    final static Logger logger = LoggerFactory.getLogger(CNNTang2015dl4j.class);

    public static final String DEFAULT_EMBEDDINGS = "glove.6B.50d";
    public static final String DEFAULT_DATASET = "lorenzo.tweet";

    protected WordVectors embeddings;
    protected LabeledSentences dataset;
    protected AnnotationPipeline pipeline;
    protected MultiLayerNetwork net;
    protected List<IterationListener> listeners = new ArrayList<>();

    //Filled with unknown words encountered when debug is turned on
    private Set<String> unknownWords = new HashSet<>();

    /**
     *
     * @param embeddings Pre-computed word-embeddings
     */
    @Inject
    public CNNTang2015dl4j(WordVectors embeddings, LabeledSentences dataset, AnnotationPipeline pipeline) throws Exception {
        this.embeddings = embeddings;
        this.dataset = dataset;
        this.pipeline = pipeline;
        buildNeuralNet(embeddings.getDim(), embeddings.getDim()*2);
    }

    private void buildNeuralNet(int wordDim, int lookupDim) throws Exception {
        int numRows = 1;
        int numColumns = 50;
        int outputNum = 5;
        int iterations = 20;
        int seed = 123;
        int[] kernel = new int[] {1, 2};
        int[] stride = new int[] {1, 1};

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(kernel, stride)
                        .nIn(wordDim*50)
                        .nOut(lookupDim)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, kernel, stride)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder, numRows, numColumns, wordDim);

        MultiLayerConfiguration conf = builder.build();

        net = new MultiLayerNetwork(conf);
        net.init();
    }

    public void addListener(IterationListener listener) {
        listeners.add(listener);
    }

    public void train() {
        //At least adding a score iteration listener
        if (listeners.size() == 0) {
            listeners.add(new ScoreIterationListener(2));
        }
        net.setListeners(listeners);
        LabeledSentences.Sentence sentence;
        int counter = 0;
        int batchSize = 100;
        ArrayList<DataSet> accumulator = new ArrayList<>();

        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
        Stopwatch watch = Stopwatch.start();
        while ((sentence = dataset.readNext()) != null) {
            int label = Integer.parseInt(sentence.label);
            int rawLabels[] = new int[5];
            rawLabels[label] = 1;

            ArrayList<INDArray> vectors = sentence2vec(sentence);
            int dim = vectors.get(0).columns();
            double[] rawInput = new double[dim*50];
            for (int i = 0; i < vectors.size() && i < 50; i++) {
                for (int j = 0; j < dim; j++) {
                    rawInput[j+i*dim] = vectors.get(i).getDouble(j);
                }
            }

            accumulator.add(new DataSet(Nd4j.create(rawInput), ArrayUtil.toNDArray(rawLabels)));
            counter++;
            if (counter % batchSize == 0) {
                logger.info(counter+" sentences processed. "+((double)watch.click()/1000)+"s");
            }
            if (counter % batchSize == 0) {
                SplitTestAndTrain split = DataSet.merge(accumulator).splitTestAndTrain(0.8);
                testInput.add(split.getTest().getFeatureMatrix());
                testLabels.add(split.getTest().getLabels());
                net.fit(split.getTrain());
                logger.info("A batch has been trained. "+((double)watch.click()/1000)+"s");
                accumulator.clear();
            }
        }

        logger.info("Evaluate model....");
        Evaluation eval = new Evaluation(5);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = net.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        logger.info(eval.stats());
    }

    /**
     * This method is used to debug the embeddings lookup layer
     * Resolves embeddings for each word and dumps the sentence model to target file
     *
     * @param target Target file
     */
    public void dumpSentenceModel(File target) {
        LabeledSentences.Sentence sentence;
        int counter = 0;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(target))) {
            while ((sentence = dataset.readNext()) != null) {
                ArrayList<INDArray> vectors = sentence2vec(sentence);
                writer.write(sentence.label);
                for (INDArray vector : vectors) {
                    for (int i = 0; i < vector.columns(); i++) {
                        writer.write(' ');
                        writer.write(Double.toString(vector.getDouble(i)));
                    }
                }
                writer.newLine();

                if (++counter % 1000 == 0) {
                    logger.info(counter+" sentences processed");
                }
            }
        } catch (IOException e) {
            logger.error("Can't open the target file", e);
        }
    }

    /**
     * Output the unknown words that system has encountered during analysis
     */
    public void outputUnknownWords() {
        //Output unknown words
        if (logger.isDebugEnabled()) {
            StringBuilder builder = new StringBuilder();
            for (String word : unknownWords) {
                if (builder.length() > 0) {
                    builder.append(", ");
                }
                builder.append(word);
            }
            logger.debug("Unknown words: "+builder.toString());
        }
    }

    /**
     * Annotate sentence
     * @param sentence Sentence as returned by the dataset
     * @return Vector representation of the sentence
     */
    private ArrayList<INDArray> sentence2vec(LabeledSentences.Sentence sentence) {
        Annotation annotation = new Annotation(sentence.sentence);
        pipeline.annotate(annotation);
        ArrayList<INDArray> result = new ArrayList<>();
        for (CoreLabel token : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
            INDArray vector = embeddings.lookup(token.word());
            result.add(vector);
            if (logger.isDebugEnabled() && embeddings.isZeroes(vector)) {
                unknownWords.add(token.word());
            }
        }
        return result;
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
        CNNTang2015dl4j project = injector.getInstance(CNNTang2015dl4j.class);

        if (params.dumpModel) {
            //Check if target folder is writable and create it if needed
            File targetFolder = new File(params.targetFolder);
            if (!targetFolder.exists() && !targetFolder.mkdirs()) {
                logger.info("Unable to create target folder. Halting");
            }

            project.dumpSentenceModel(new File(targetFolder, params.sentencesFilename));
            project.outputUnknownWords();
        } else {
            if (params.enableStatistics) {
                try {
                    UiServer.main(null);
                } catch (Exception e) {
                    logger.error("Can't start webserver", e);
                }
                project.addListener(new HistogramIterationListener(2));
            }
            project.train();
        }
    }
}
