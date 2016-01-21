package eu.fbk.hlt.sentiment.baseline;

import java.io.*;
import java.util.*;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.Tree;
import libsvm.*;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

public class NBSVM {
    private static StanfordCoreNLP pipeline = null;

    //global variables
    //all the sentences
    //private static ArrayList<String> trainingSentences = new ArrayList<>();

    //all the labels
    private static ArrayList<String> trainingLabels = new ArrayList<>();

    //maps name of class to array idx
    private static HashMap<String, Integer> classIdx = new HashMap<>(4);

    //"matrix" of sentence features (ratio of the words contained in a sentence)
    private static ArrayList<Vector<Integer>> sentencesFeatures = new ArrayList<>();

    //global dictionary: for each ngram how many times it appears
    private static HashMap<String, Integer> ngramsDict = new HashMap<>();

    //counter for each class
    private static ArrayList<HashMap<String, Integer>> counters = new ArrayList<>();

    //
    private static HashMap<String,Integer> indexedDict = new HashMap<>();

    //grams to use (e.g. {1, 3} for unigrams + trigrams)
    private static ArrayList<Integer> ngramsToUse = null;

    private static ArrayList<Vector<Double>> ratios = new ArrayList<>();

    private static int lastClass = 0;

    //read file and create dictionaries
    private static void readFile(String fileName) throws Exception {
        FileInputStream fIn = new FileInputStream(fileName);
        BufferedReader in = new BufferedReader(new InputStreamReader(fIn));
        String line;

        ArrayList<String> fileContent = new ArrayList<>();
        while ((line = in.readLine()) != null) {
            fileContent.add(line);
        }
        in.close();
        fIn.close();
        Collections.sort(fileContent);

        lastClass = 0;
        int wordIdx = 0;
        HashMap<String, Integer> classCounter = null;
        for (String currLine : fileContent) {

            String[] content = currLine.split("\t");
            trainingLabels.add(content[0]);
            //trainingSentences.add(content[1]);
            //if we never saw this label before, give it an index and initialize its dict after saving the old
            if (!classIdx.containsKey(content[0])) {
                classIdx.put(content[0], lastClass);

                lastClass++;
                if (classCounter != null)
                    counters.add(classCounter);
                classCounter = new HashMap<>();
            }

            //tokenize text
            Annotation annotation = new Annotation(content[1].trim());
            pipeline.annotate(annotation);

            Vector<Integer> sentenceFeature = new Vector<>(annotation.get(CoreAnnotations.TokensAnnotation.class).size());

            //create all requested n-grams
            for (int n : ngramsToUse) {
                int i = 0;
                List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
                for (CoreLabel token : tokens) {
                    if (i <= tokens.size() - n) {
                        StringBuilder sb = new StringBuilder();
                        for (int j = 0; j < n - 1; j++) {
                            sb.append(tokens.get(i + j).originalText().toLowerCase()).append(" ");
                        }
                        sb.append(tokens.get(i + n - 1).originalText().toLowerCase());
                        String gram = sb.toString();
                        //update count in current counter
                        if (classCounter.containsKey(gram))
                            classCounter.put(gram, classCounter.get(gram) + 1);
                        else classCounter.put(gram, 1);

                        int thisGramIdx;
                        //assign an index value to this ngram and update the global count
                        if (!indexedDict.containsKey(gram)) {
                            indexedDict.put(gram,wordIdx);
                            thisGramIdx = wordIdx;
                            wordIdx++;
                            ngramsDict.put(gram,1);
                        }
                        else {
                            ngramsDict.put(gram, ngramsDict.get(gram) + 1);
                            thisGramIdx = indexedDict.get(gram);
                        }
                        sentenceFeature.add(thisGramIdx);
                    }
                    i++;
                }
            }
            sentencesFeatures.add(sentenceFeature);
        }
        counters.add(classCounter);
    }



    public static void main(String[] args) throws Exception {
        String trainFileName;
        String testFileName;
        String outFileName;

        ArgumentParser parser = ArgumentParsers.newArgumentParser("NBSVM").description("Java conversion of https://github.com/mesnilgr/nbsvm");
        parser.addArgument("-train").type(String.class).help("the training file");
        parser.addArgument("-test").type(String.class).help("the test file");
        parser.addArgument("-out").type(String.class).help("the output file");
        parser.addArgument("-ngrams").type(Integer.class).nargs("+").help("the grams to use (e.g. 1 2 3 uses uni+bi+trigrams");

        try {
            Namespace res = parser.parseArgs(args);
            trainFileName = res.get("train");
            testFileName = res.get("-test");
            outFileName = res.get("-out");
            ngramsToUse = res.get("ngrams");
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            return;
        }


        Properties props = new Properties();
        props.put("annotators", "tokenize");
        pipeline = new StanfordCoreNLP(props);

        readFile(trainFileName);
        computeRatios();
        trainSVM();
        return;
    }


    private static void computeRatios() {
        for (int c = 0; c < lastClass; c++) { //C++ is evil
            //TODO: convert pClass in a Double[]
            Vector <Double> pClass = new Vector<>(ngramsDict.size());
            for (int i=0; i < ngramsDict.size(); i++)
                pClass.add(1d);
            HashMap<String, Integer> classCounter = counters.get(c);
            double sumOfArray = 0;
            for (String word : ngramsDict.keySet()) {
                int wordIdx = ngramsDict.get(word);
                double countForClass = 0;
                if (classCounter.containsKey(word)) {
                    countForClass += classCounter.get(word).doubleValue();
                }
                pClass.set(wordIdx,countForClass+1);
                sumOfArray += countForClass+1;
            }
            for (int i=0; i < ngramsDict.size(); i++) {
                Double prob = pClass.get(i)/sumOfArray;
                pClass.set(i,Math.log(prob/(1-prob)));
            }

            ratios.add(pClass);

        }
    }


    private static void trainSVM() {
        Vector<Double> vy = new Vector<>();
        Vector<svm_node[]> vx = new Vector<>();

        int max_index = 0;
        for (int i = 0; i < trainingLabels.size(); i++) {
            Integer classNum = classIdx.get(trainingLabels.get(i));
            vy.add(classNum.doubleValue());
            //Vector<Integer> features = sentencesFeatures.get(i);

            int m = sentencesFeatures.get(i).size();
            svm_node[] x = new svm_node[m];
            int j = 0;
            for (Integer word : sentencesFeatures.get(i)) {
                x[j] = new svm_node();
                x[j].index = word;
                x[j].value = ratios.get(classNum).get(word);
                j++;
            }
            if(m>0) max_index = Math.max(max_index, x[m-1].index);
            vx.addElement(x);
        }


        svm_problem prob = new svm_problem();
        prob.l = vy.size();
        prob.x = new svm_node[prob.l][];
        for(int i=0;i<prob.l;i++)
            prob.x[i] = vx.elementAt(i);
        prob.y = new double[prob.l];
        for(int i=0;i<prob.l;i++)
            prob.y[i] = vy.elementAt(i);

        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.degree = 3;
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];

        if(max_index > 0)
            param.gamma = 1.0/max_index;
        System.out.println("training SVM model");
        long tStart = System.currentTimeMillis();
        svm_model model = svm.svm_train(prob,param);
        long tEnd = System.currentTimeMillis();
        double elapsedSeconds = (tEnd-tStart) / 1000.0;
        System.out.println("finished training in "+elapsedSeconds+ " sec.");

    }
}