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
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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
public class CNNTang2015dl4j {
    final static Logger logger = LoggerFactory.getLogger(CNNTang2015dl4j.class);

    public static final String EMBEDDINGS = "glove.6B.50d";
    public static final String DATASET = "lorenzo.amazon";

    protected WordVectors embeddings;
    protected LabeledSentences dataset;
    protected AnnotationPipeline pipeline;
    protected MultiLayerNetwork net;

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
        buildNeuralNet(embeddings.getDim(), embeddings.getDim());
    }

    private void buildNeuralNet(int wordDim, int lookupDim) throws Exception {
        int numRows = 50;
        int numColumns = 1;
        int outputNum = 5;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = iterations/5;

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(1, 1)
                        .stride(1,1)
                        .nIn(wordDim)
                        .nOut(lookupDim)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] {1,1})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,wordDim);

        MultiLayerConfiguration conf = builder.build();

        net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));
    }

    public void train() {
        LabeledSentences.Sentence sentence;
        int counter = 0;
        ArrayList<DataSet> accumulator = new ArrayList<>();

        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();
        while ((sentence = dataset.readNext()) != null) {
            int label = Integer.parseInt(sentence.label);
            int rawLabels[] = new int[5];
            rawLabels[label] = 1;

            ArrayList<SimpleMatrix> vectors = sentence2vec(sentence);
            int dim = vectors.get(0).numRows();
            double[] rawInput = new double[dim*50];
            for (int i = 0; i < vectors.size() && i < 50; i++) {
                for (int j = 0; j < dim; j++) {
                    rawInput[j+i*dim] = vectors.get(i).get(j, 0);
                }
            }

            accumulator.add(new DataSet(Nd4j.create(rawInput), ArrayUtil.toNDArray(rawLabels)));
            if (++counter % 100 == 0) {
                SplitTestAndTrain split = DataSet.merge(accumulator).splitTestAndTrain(90);
                testInput.add(split.getTest().getFeatureMatrix());
                testLabels.add(split.getTest().getLabels());
                net.fit(split.getTrain());
                logger.info(counter+" sentences processed");
                accumulator.clear();
            }
        }

        logger.info("Evaluate model....");
        Evaluation eval = new Evaluation(5);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = net.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = net.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
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
                ArrayList<SimpleMatrix> vectors = sentence2vec(sentence);
                writer.write(sentence.label);
                for (SimpleMatrix vector : vectors) {
                    for (int i = 0; i < vector.numRows(); i++) {
                        writer.write(' ');
                        writer.write(Double.toString(vector.get(i, 0)));
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
    private ArrayList<SimpleMatrix> sentence2vec(LabeledSentences.Sentence sentence) {
        Annotation annotation = new Annotation(sentence.sentence);
        pipeline.annotate(annotation);
        ArrayList<SimpleMatrix> result = new ArrayList<>();
        for (CoreLabel token : annotation.get(CoreAnnotations.TokensAnnotation.class)) {
            SimpleMatrix vector = embeddings.lookup(token.word());
            result.add(vector);
            if (logger.isDebugEnabled() && embeddings.isZeroes(vector)) {
                unknownWords.add(token.word());
            }
        }
        return result;
    }

    /**
     * The most magical method I've ever written. Magic is in every line.
     *
     * Just kidding. Here we automatically bootstrap
     *  our environment with just a little bit
     *  of a custom WordVectors construction
     *  via the DatasetProvider class
     */
    public static void main(String[] args) throws ParseException {
        //Parameters params = new Parameters(args);
        Injector injector = Guice.createInjector(new DatasetProvider());
        CNNTang2015dl4j project = injector.getInstance(CNNTang2015dl4j.class);
        //project.dumpSentenceModel(new File(params.target));
        //project.outputUnknownWords();
        project.train();
    }

    public static class DatasetProvider extends AbstractModule {
        @Override
        protected void configure() {}

        @Provides
        AnnotationPipeline providePipeline() {
            Properties commonProps = new Properties();
            commonProps.setProperty("annotators", "tokenize, ssplit, parse");
            StanfordCoreNLP pipeline = new StanfordCoreNLP(commonProps);
            BinarizerAnnotator binarizerAnnotator = new BinarizerAnnotator("ba", new Properties());
            pipeline.addAnnotator(binarizerAnnotator);
            return pipeline;
        }

        @Provides
        WordVectors provideWordVectors(DatasetRepository repository) throws Exception {
            Dataset dataset = repository.load(EMBEDDINGS);
            if (!(dataset instanceof WordVectors)) {
                throw new Exception("The instantiated dataset is of the wrong type");
            }
            return (WordVectors) dataset;
        }

        @Provides
        LabeledSentences provideDataset(DatasetRepository repository) throws Exception {
            Dataset dataset = repository.load(DATASET);
            if (!(dataset instanceof LabeledSentences)) {
                throw new Exception("The instantiated dataset is of the wrong type");
            }
            return (LabeledSentences) dataset;
        }
    }

    public static class Parameters {
        public String target;

        public Parameters(String[] args) throws ParseException {
            //Defining input parameters
            Options options = new Options();
            CLIOptionBuilder builder = new CLIOptionBuilder().hasArg().withArgName("file").isRequired();

            options.addOption(builder.withDescription("Where to output the results of lookup layer").withLongOpt("target").toOption("t"));

            //Parsing the input
            CommandLineParser parser = new PosixParser();
            CommandLine line;
            try {
                //Parse the command line arguments
                line = parser.parse(options, args);

                //Filling the initial configuration
                target = line.getOptionValue("target");
            } catch (ParseException e) {
                //If parameters are wrong — print help
                new HelpFormatter().printHelp(400, "java -Dfile.encoding=UTF-8 " + CNNTang2015dl4j.class.getName(), "\n", options, "\n", true);
                throw e;
            }
        }
    }
}
