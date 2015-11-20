package eu.fbk.hlt.sentiment.baseline;

import com.google.inject.AbstractModule;
import com.google.inject.Guice;
import com.google.inject.Injector;
import com.google.inject.Provides;
import eu.fbk.hlt.data.Dataset;
import eu.fbk.hlt.data.DatasetRepository;
import eu.fbk.hlt.data.WordVectors;

import javax.inject.Inject;

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
    //TODO: impossible to do with deeplearning4j -> no merge layers. Either find different library or implement necessary layers yourself
    protected WordVectors embeddings;

    /**
     *
     * @param embeddings Pre-computed word-embeddings
     */
    @Inject
    public CNNTang2015(WordVectors embeddings) {
        this.embeddings = embeddings;
    }

    public void start() {

    }

    /**
     * The most magical method I've ever written. Magic is in every line.
     *
     * Just kidding. Here we automatically bootstrap
     *  our environment with just a little bit
     *  of a custom WordVectors construction
     *  via the DatasetProvider class
     */
    public static void main(String[] args) {
        Injector injector = Guice.createInjector(new DatasetProvider());
        CNNTang2015 project = injector.getInstance(CNNTang2015.class);
        project.start();
    }

    public static class DatasetProvider extends AbstractModule {
        @Override
        protected void configure() {}

        @Provides
        WordVectors provideWordVectors(DatasetRepository repository) throws Exception {
            Dataset dataset = repository.load("glove.6B.50d");
            if (!(dataset instanceof WordVectors)) {
                throw new Exception("The instantiated dataset is of the wrong type");
            }
            return (WordVectors) dataset;
        }
    }
}
