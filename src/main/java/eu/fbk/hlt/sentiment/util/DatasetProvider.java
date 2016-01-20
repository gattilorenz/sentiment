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
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * A module that resolves core dependencies for the models
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class DatasetProvider extends AbstractModule {
    protected String dataset;
    protected String embeddings;

    public DatasetProvider(SentimentParameters params) {
        dataset = params.dataset;
        embeddings = params.embeddings;
    }

    @Override
    protected void configure() {
        ReflectionsHelper.registerUrlTypes();
    }

    @Provides
    @Named("classes")
    List<String> provideClasses() {
        return new ArrayList<String>() {{
            add("0");
            add("1");
            add("2");
            add("3");
            add("4");
        }};
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
        return (LabeledSentences) dataset;
    }
}