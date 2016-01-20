package eu.fbk.hlt.sentiment;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import eu.fbk.hlt.data.LabeledSentences;
import eu.fbk.hlt.data.WordVectors;
import eu.fbk.hlt.sentiment.util.Stopwatch;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A generic model with training and evaluation step
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public abstract class AbstractModel {
    protected final static Logger logger = LoggerFactory.getLogger(AbstractModel.class);

    public static final int DEFAULT_BATCH_SIZE = 100;
    //Save 20% of samples randomly as a test set
    public static final double DEFAULT_TEST_SPLIT = 0.2;

    protected WordVectors embeddings;
    protected AnnotationPipeline pipeline;
    protected List<String> classes;

    List<SentenceModelListener> sentenceModelListeners = new ArrayList<>();

    public AbstractModel(List<String> classes, WordVectors embeddings, AnnotationPipeline pipeline) {
        this.embeddings = embeddings;
        this.pipeline = pipeline;
        this.classes = classes;
    }

    /**
     * Trains the model using the provided dataset
     */
    public void train(LabeledSentences dataset) {
        int counter = 0;
        List<LabeledSentences.Sentence> testInput = new ArrayList<>();
        Stopwatch watch = Stopwatch.start();
        Random testRnd = new Random();
        LabeledSentences.Sentence sentence;
        logger.info("Starting training with dataset \""+dataset.getInfo().name+"\"");
        while ((sentence = dataset.readNext()) != null) {
            //Add some of the samples to test set
            if (testRnd.nextDouble() <= DEFAULT_TEST_SPLIT) {
                testInput.add(sentence);
                continue;
            }
            INDArray input = sentence2mat(sentence);
            INDArray label = getLabelVector(sentence.label);
            train(input, label);

            counter++;
            if (counter % DEFAULT_BATCH_SIZE == 0) {
                logger.info(counter+" sentences processed. "+((double)watch.click()/1000)+"s");
            }
        }

        logger.info("Evaluate model....");
        logger.info(evaluate(testInput).stats());
    }

    /**
     * Predict a label of a sentence (output a vector of probabilities for each class)
     */
    public INDArray predict(LabeledSentences.Sentence sentence) {
        return predict(sentence2mat(sentence));
    }

    /**
     * Feed a single training sample to the model
     */
    protected abstract void train(INDArray input, INDArray label);

    protected abstract INDArray predict(INDArray input);

    public Evaluation evaluate(List<LabeledSentences.Sentence> test) {
        Evaluation eval = new Evaluation(classes);
        for (LabeledSentences.Sentence sentence : test) {
            eval.eval(getLabelVector(sentence.label), predict(sentence));
        }
        return eval;
    }

    public INDArray getLabelVector(String label) {
        INDArray vector = Nd4j.zeros(classes.size());
        vector.putScalar(getLabelIndex(label), 1.0);
        return vector;
    }

    public int getLabelIndex(String label) {
        for (int i = 0; i < classes.size(); i++) {
            if (label.equals(classes.get(i))) {
                return i;
            }
        }
        logger.warn("Label \""+label+"\" was not found. Results may be invalid!");
        return 0;
    }

    /**
     * Annotate sentence
     * @param sentence Sentence as returned by the dataset
     * @return Matrix representation of the sentence
     */
    private INDArray sentence2mat(LabeledSentences.Sentence sentence) {
        int dim = embeddings.getDim();
        Annotation annotation = new Annotation(sentence.sentence);
        pipeline.annotate(annotation);
        //ArrayList<INDArray> result = new ArrayList<>();
        List<CoreLabel> annotations = annotation.get(CoreAnnotations.TokensAnnotation.class);
        INDArray result = Nd4j.create(dim*annotations.size());
        int counter = 0;
        for (CoreLabel token : annotations) {
            INDArray vector = embeddings.lookup(token.word().toLowerCase());
            result.put(new INDArrayIndex[]{NDArrayIndex.interval(counter*dim, (counter+1)*dim)}, vector);
            counter++;
        }
        INDArray matResult = result.reshape(annotations.size(), dim);
        //Notify listeners
        sentenceModelListeners.forEach(value -> value.process(sentence.label, matResult));
        return matResult;
    }

    public void addSentenceModelListener(SentenceModelListener listener) {
        sentenceModelListeners.add(listener);
    }

    public interface SentenceModelListener {
        public void process(String label, INDArray sentence);
    }
}
