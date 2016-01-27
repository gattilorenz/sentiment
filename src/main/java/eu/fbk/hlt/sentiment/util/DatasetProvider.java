package eu.fbk.hlt.sentiment.util;

import com.google.inject.AbstractModule;
import com.google.inject.Provides;
import com.google.inject.name.Named;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.pipeline.BinarizerAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import eu.fbk.hlt.data.Dataset;
import eu.fbk.hlt.data.DatasetRepository;
import eu.fbk.hlt.data.LabeledSentences;
import eu.fbk.hlt.data.WordVectors;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.*;

/**
 * A module that resolves core dependencies for the models
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class DatasetProvider extends AbstractModule {
    protected final String dataset;
    protected final String embeddings;
    protected final boolean threeClass;

    public DatasetProvider(SentimentParameters params) {
        dataset = params.dataset;
        embeddings = params.embeddings;
        threeClass = params.threeClass;
    }

    @Override
    protected void configure() {
        ReflectionsHelper.registerUrlTypes();
    }

    @Provides
    @Named("classes")
    List<String> provideClasses() {
        List<String> classes = new ArrayList<>();
        classes.add("1");
        classes.add("2");
        classes.add("3");
        if (!threeClass) {
            classes.add("0");
            classes.add("4");
        }
        return classes;
    }

    @Provides
    AnnotationPipeline providePipeline() {
        //Silence output to err
        System.setErr(new PrintStream(new OutputStream() {public void write(int b) {}}));

        Properties commonProps = new Properties();
        commonProps.setProperty("annotators", "tokenize, ssplit, parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(commonProps);
        BinarizerAnnotator binarizerAnnotator = new BinarizerAnnotator("ba", new Properties());
        pipeline.addAnnotator(binarizerAnnotator);

        //Restore output to err
        System.setErr(System.err);
        return pipeline;
    }

    @Provides
    WordVectors provideWordVectors(DatasetRepository repository) throws Exception {
        Dataset dataset = repository.load(this.embeddings);
        if (!(dataset instanceof WordVectors)) {
            throw new Exception("The instantiated dataset is of the wrong type");
        }
        return (WordVectors) dataset;
    }

    @Provides
    LabeledSentences provideDataset(DatasetRepository repository) throws Exception {
        Dataset dataset = repository.load(this.dataset);
        if (!(dataset instanceof LabeledSentences)) {
            throw new Exception("The instantiated dataset is of the wrong type");
        }
        //Remap dataset to three classes
        if (threeClass) {
            Map<String, String> mappings = new HashMap<>();
            mappings.put("0", "1");
            mappings.put("4", "3");
            dataset = new LabeledSentences.RemappedLabeledSentences(mappings, dataset.getInfo());
        }
        return (LabeledSentences) dataset;
    }
}