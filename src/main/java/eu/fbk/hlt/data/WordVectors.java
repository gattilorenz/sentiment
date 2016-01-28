package eu.fbk.hlt.data;

import com.google.inject.Guice;
import com.google.inject.Injector;
import edu.stanford.nlp.util.Generics;
import eu.fbk.hlt.sentiment.util.DatasetProvider;
import eu.fbk.hlt.sentiment.util.SentimentParameters;
import eu.fbk.hlt.sentiment.util.Stopwatch;
import org.apache.commons.cli.ParseException;
import org.apache.commons.collections.SortedBag;
import org.ejml.simple.SimpleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URISyntaxException;
import java.util.*;

/**
 * A word vector lookup map, mainly used to store word embeddings
 * The format is
 *  WORD X1 X2 X3 X4
 *
 * Optionally can perform L2 normalization
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class WordVectors extends Dataset {
    final static Logger logger = LoggerFactory.getLogger(WordVectors.class);

    protected Map<String, INDArray> wordVectors;
    protected INDArray zeroes;
    protected int dim;

    public WordVectors(DatasetMetaInfo info) throws URISyntaxException {
        super(info);
    }

    @Override
    public void parse() {
        //Parse the input file
        wordVectors = Generics.newHashMap();
        try (LineNumberReader reader = getReader()) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] elements = line.split("\\s+");
                //Sanitizing the the word
                String word = elements[0];
                if(word.equals("UNKNOWN") || word.equals("UUUNKKK") || word.equals("UNK") || word.equals("*UNKNOWN*") || word.equals("<unk>")) {
                    word = "*UNK*";
                }

                if(word.equals("<s>")) {
                    word = "*START*";
                }

                if(word.equals("</s>")) {
                    word = "*END*";
                }

                //Checking dimensions
                if (dim == 0) {
                    dim = elements.length - 1;
                }
                if (dim >= elements.length) {
                    logger.warn("Skipping the word \""+word+"\" — vector is too small. Got: "+(elements.length-1)+". Need: "+dim);
                    continue;
                }

                //Populating word vector
                double[] vector = new double[dim];
                for (int i = 0; i < dim; i++) {
                    vector[i] = Double.valueOf(elements[i+1]);
                }

                //Saving into the dictionary
                wordVectors.put(word, Nd4j.create(vector));
            }

            zeroes = Nd4j.zeros(dim);
            logger.info("Parsed "+wordVectors.size()+" words");
        } catch (IOException e) {
            logger.error("Can't parse the input file: "+e.getClass().getSimpleName()+" "+e.getMessage());
        }
    }

    public Map<Double, String> findSimilar(String word) {
        INDArray initial = wordVectors.getOrDefault(word, zeroes);
        TreeMap<Double, String> scores = new TreeMap<>();
        Iterator<Map.Entry<String, INDArray>> iterator = wordVectors.entrySet().iterator();
        int counter = 0;
        while (iterator.hasNext()) {
            Map.Entry<String, INDArray> entry = iterator.next();
            INDArray value = entry.getValue();
            double similarity = Transforms.cosineSim(value, initial);
            if (scores.size() < 10 || similarity > scores.firstKey()) {
                scores.put(similarity, entry.getKey());
                if (scores.size() == 11) {
                    scores.remove(scores.firstKey());
                }
            }
            if (++counter % 100000 == 0) {
                logger.debug("Looked up "+counter+" words");
            }
        }
        return scores.descendingMap();
    }

    public INDArray lookup(String word) {
        return wordVectors.getOrDefault(word, zeroes);
    }

    public boolean isZeroes(INDArray vector) {
        return vector.equals(zeroes);
    }

    public int getDim() {
        return dim;
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
        WordVectors wordVectors = injector.getInstance(WordVectors.class);
        wordVectors.interactive();
    }

    public void interactive() {
        logger.info("Enabling interactive mode");

        try (BufferedReader input = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.println("Please write a token:");
                Stopwatch watch = Stopwatch.start();
                String token = input.readLine().toLowerCase();
                Map<Double, String> result = this.findSimilar(token);
                System.out.println("Found similar tokens for \""+token+"\" in "+watch.click()+" ms");
                System.out.println("Similar:");
                for (Map.Entry<Double, String> score : result.entrySet()) {
                    System.out.println(" "+score.getValue()+" — "+score.getKey());
                }
                System.out.println();
            }
        } catch (IOException e) {
            logger.error("Error while reading from stream or closing", e);
        }
    }
}
