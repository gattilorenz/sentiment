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
import eu.fbk.hlt.sentiment.nn.Pipeline;
import eu.fbk.hlt.sentiment.nn.duyu.LinearLayer;
import eu.fbk.hlt.sentiment.nn.duyu.MultiConnectLayer;
import eu.fbk.hlt.sentiment.nn.duyu.TanhLayer;
import eu.fbk.hlt.sentiment.util.CLIOptionBuilder;
import org.apache.commons.cli.*;
import org.ejml.simple.SimpleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.inject.Inject;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

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
public class CNNTang2015 {
    final static Logger logger = LoggerFactory.getLogger(CNNTang2015.class);

    public static final String EMBEDDINGS = "glove.6B.50d";
    public static final String DATASET = "lorenzo.amazon";

    //TODO: impossible to do with deeplearning4j -> no merge layers. Either find different library or implement necessary layers yourself
    protected WordVectors embeddings;
    protected LabeledSentences dataset;
    protected AnnotationPipeline pipeline;

    //Filled with unknown words encountered when debug is turned on
    private Set<String> unknownWords = new HashSet<>();

    /**
     *
     * @param embeddings Pre-computed word-embeddings
     */
    @Inject
    public CNNTang2015(WordVectors embeddings, LabeledSentences dataset, AnnotationPipeline pipeline) {
        this.embeddings = embeddings;
        this.dataset = dataset;
        this.pipeline = pipeline;
    }

    private Pipeline buildNeuralNet(int wordDim, int lookupDim) throws Exception {
        //Create a set of layers for each of the window sizes
        //Pipeline
        for (int windowSize = 1; windowSize < 3; windowSize++) {
            LinearLayer linear = new LinearLayer(wordDim * windowSize, lookupDim);
            TanhLayer tanh = new TanhLayer(lookupDim);
            Pipeline filter = new Pipeline(linear);
            filter.after(tanh);
        }
        return null;
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
        Parameters params = new Parameters(args);
        Injector injector = Guice.createInjector(new DatasetProvider());
        CNNTang2015 project = injector.getInstance(CNNTang2015.class);
        project.dumpSentenceModel(new File(params.target));
        project.outputUnknownWords();
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
                new HelpFormatter().printHelp(400, "java -Dfile.encoding=UTF-8 " + CNNTang2015.class.getName(), "\n", options, "\n", true);
                throw e;
            }
        }
    }
}
