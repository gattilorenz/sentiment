package eu.fbk.hlt.data;

import edu.stanford.nlp.util.Generics;
import org.ejml.simple.SimpleMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URISyntaxException;
import java.util.Map;

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

    public INDArray lookup(String word) {
        return wordVectors.getOrDefault(word, zeroes);
    }

    public boolean isZeroes(INDArray vector) {
        return vector.equals(zeroes);
    }

    public int getDim() {
        return dim;
    }
}
