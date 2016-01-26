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
    private static ArrayList<String> trainLabels = new ArrayList<>();
    private static ArrayList<String> testLabels = new ArrayList<>();

    //maps name of class to array idx
    private static HashMap<String, Integer> classIdx = new HashMap<>(4);

    //"matrix" of sentence features (ratio of the words contained in a sentence)
    private static ArrayList<Vector<Integer>> trainFeatures = new ArrayList<>();
    private static ArrayList<Vector<Integer>> testFeatures = new ArrayList<>();

    //global dictionary: for each ngram how many times it appears
    private static HashMap<String, Integer> ngramsDict = new HashMap<>();

    //counter for each class
    private static HashMap<String, HashMap<String, Integer>> counters = new HashMap<>();

    //
    private static HashMap<String, Integer> indexedDict = new HashMap<>();

    //grams to use (e.g. {1, 3} for unigrams + trigrams)
    private static ArrayList<Integer> ngramsToUse = null;

    private static ArrayList<Vector<Double>> ratios = new ArrayList<>();

    private static int lastClass = 0;

    private static svm_model model;

    private static int splitPercentage = 60;

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
        Collections.shuffle(fileContent);

        int trainingSize = (int) fileContent.size() * splitPercentage / 100;

        int currWordIdx = 0;
        HashMap<String, Integer> classCounter = null;

        for (int i = 0; i < fileContent.size(); i++) {
            String currLine = fileContent.get(i);
            if (i < trainingSize)
                currWordIdx = processLine(currLine, trainLabels, trainFeatures, true, currWordIdx);
            else
                currWordIdx = processLine(currLine, testLabels, testFeatures, false, currWordIdx);
        }
    }

    //returns the last word idx
    private static int processLine(String line, ArrayList<String> labels, ArrayList<Vector<Integer>> features,
                                   boolean training, int currWordIdx) {
        String[] content = line.split("\t");
        labels.add(content[0]);

        HashMap<String, Integer> classCounter = null;

        //if we never saw this label before, give it an index and initialize its dict
        if (!classIdx.containsKey(content[0])) {
            classIdx.put(content[0], lastClass);
            lastClass++;
            classCounter = new HashMap<>();
            counters.put(content[0], classCounter);
        } else {
            classCounter = counters.get(content[0]);
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
                    if (training) {
                        if (classCounter.containsKey(gram))
                            classCounter.put(gram, classCounter.get(gram) + 1);
                        else classCounter.put(gram, 1);
                    }

                    int thisGramIdx;
                    //assign an index value to this ngram and update the global count
                    if (!indexedDict.containsKey(gram)) {
                        indexedDict.put(gram, currWordIdx);
                        thisGramIdx = currWordIdx;
                        currWordIdx++;
                        ngramsDict.put(gram, 1);
                    } else {
                        ngramsDict.put(gram, ngramsDict.get(gram) + 1);
                        thisGramIdx = indexedDict.get(gram);
                    }
                    sentenceFeature.add(thisGramIdx);
                }
                i++;
            }
        }
        features.add(sentenceFeature);
        return currWordIdx;
    }


    public static void main(String[] args) throws Exception {
        String dataFileName = "";
        String outFileName;
        String svmModelFile = "";

        ngramsToUse = new ArrayList<>(3);
        ngramsToUse.add(1);
        ngramsToUse.add(2);
        ngramsToUse.add(3);

        ArgumentParser parser = ArgumentParsers.newArgumentParser("NBSVM").description("Java conversion of https://github.com/mesnilgr/nbsvm");
        parser.addArgument("-data").type(String.class).help("tsv input file (format: label\\tsentence)");
        parser.addArgument("-model").type(String.class).help("where to load/save the model");
        parser.addArgument("-out").type(String.class).help("output file with predictions and real labels");
        parser.addArgument("-ngrams").type(Integer.class).nargs("+").help("the grams to use (e.g. 1 2 3 uses uni+bi+trigrams, the default");
        parser.addArgument("-split").type(Integer.class).help("what percentage to use for the split (default: 60)");
        try {
            Namespace res = parser.parseArgs(args);
            dataFileName = res.get("data");
            svmModelFile = res.get("model");
            outFileName = res.get("out");
            ngramsToUse = res.get("ngrams");

        } catch (ArgumentParserException e) {
            parser.handleError(e);
            return;
        }

        if (new File(dataFileName).isFile()) {
            System.err.println("Invalid input file path, exiting.");
            return;
        }


        Properties props = new Properties();
        props.put("annotators", "tokenize");
        pipeline = new StanfordCoreNLP(props);

        readFile(dataFileName);
        computeRatios();

        if (new File(svmModelFile).isFile()) {
            System.out.println("Model file " + svmModelFile + " exists, loading it...");
            model = svm.svm_load_model(svmModelFile);
            System.out.println("Model loaded.");
        } else {
            trainSVM(svmModelFile);
        }


        predict(outFileName);
        System.out.println("Predicted everything!");
        return;
    }


    private static void computeRatios() {
        for (int c = 0; c < lastClass; c++) { //C++ is evil
            //TODO: convert pClass in a Double[]
            Vector<Double> pClass = new Vector<>(ngramsDict.size());
            for (int i = 0; i < ngramsDict.size(); i++)
                pClass.add(1d);
            HashMap<String, Integer> classCounter = counters.get(String.valueOf(c));
            if (classCounter == null) {
                for (int i = 0; i < ngramsDict.size(); i++)
                    pClass.set(i, Math.log(0d));
                ratios.add(pClass);
                continue;
            }
            double sumOfArray = 0;
            for (String word : ngramsDict.keySet()) {
                int wordIdx = ngramsDict.get(word);
                double countForClass = 0;
                if (classCounter.containsKey(word)) {
                    countForClass += classCounter.get(word).doubleValue();
                }
                pClass.set(wordIdx, countForClass + 1);
                sumOfArray += countForClass + 1;
            }
            for (int i = 0; i < ngramsDict.size(); i++) {
                Double prob = pClass.get(i) / sumOfArray;
                pClass.set(i, Math.log(prob / (1 - prob)));
            }

            ratios.add(pClass);

        }
    }


    private static void trainSVM(String svmModelFile) throws IOException {
        Vector<Double> vy = new Vector<>();
        Vector<svm_node[]> vx = new Vector<>();

        int max_index = 0;
        for (int i = 0; i < trainLabels.size(); i++) {
            Integer classNum = classIdx.get(trainLabels.get(i));
            vy.add(classNum.doubleValue());
            //Vector<Integer> features = sentencesFeatures.get(i);

            int m = trainFeatures.get(i).size();
            svm_node[] x = new svm_node[m];
            int j = 0;
            for (Integer word : trainFeatures.get(i)) {
                x[j] = new svm_node();
                x[j].index = word + 1; //features starts from 1, not 0
                x[j].value = ratios.get(classNum).get(word);
                j++;
            }
            if (m > 0) max_index = Math.max(max_index, x[m - 1].index);
            vx.addElement(x);
        }


        svm_problem prob = new svm_problem();
        prob.l = vy.size();
        prob.x = new svm_node[prob.l][];
        for (int i = 0; i < prob.l; i++)
            prob.x[i] = vx.elementAt(i);
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.elementAt(i);

        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.degree = 3;
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 35;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 0; //default: 1
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];

        if (max_index > 0)
            param.gamma = 1.0 / max_index;

        long tStart = System.currentTimeMillis();
        System.out.println("training SVM model with " + trainLabels.size() + " examples (started at " +
                Calendar.getInstance().get(Calendar.HOUR_OF_DAY) + ":" + Calendar.getInstance().get(Calendar.MINUTE) + ")");
        model = svm.svm_train(prob, param);


        int correct = 0;
        int total = 0;
        for (int i = 0; i < trainLabels.size(); i++) {
            double target = classIdx.get(trainLabels.get(i));
            svm_node[] x = prob.x[i];
            double v = svm.svm_predict(model, x);
            if (v == target)
                ++correct;
            total++;
        }
        System.out.println("Accuracy on training data = " + (double) correct / total * 100 +
                "% (" + correct + "/" + total + ") (classification)\n");

        long tEnd = System.currentTimeMillis();
        double elapsedSeconds = (tEnd - tStart) / 1000.0;
        System.out.println("finished training in " + Math.round(elapsedSeconds / 60) + " min.");
        try {
            svm.svm_save_model(svmModelFile, model);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static void predict(String outFileName) throws IOException {

        DataOutputStream output = null;
        if (outFileName != null && outFileName != "")
            output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(outFileName)));
        int correct = 0;
        int total = 0;
        double error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;

        int svm_type = svm.svm_get_svm_type(model);
        int nr_class = svm.svm_get_nr_class(model);
        double[] prob_estimates = null;
        if (output != null)
            output.writeBytes("actual\tpredicted\n");
        for (int i = 0; i < testLabels.size(); i++) {
            double target = classIdx.get(testLabels.get(i));
            //output.writeBytes(target + " ");
            int m = testFeatures.get(i).size();
            svm_node[] x = new svm_node[m];
            int j = 0;
            for (Integer word : testFeatures.get(i)) {

                x[j] = new svm_node();
                x[j].index = word + 1;
                x[j].value = ratios.get(classIdx.get(testLabels.get(i))).get(word);
                j++;
            }
            double v;
            v = svm.svm_predict(model, x);
            if (output != null)
                output.writeBytes("" + target + "\t" + v + "\n");

            if (v == target)
                ++correct;
            error += (v - target) * (v - target);
            sumv += v;
            sumy += target;
            sumvv += v * v;
            sumyy += target * target;
            sumvy += v * target;
            ++total;
        }
        System.out.println("Accuracy = " + (double) correct / total * 100 +
                "% (" + correct + "/" + total + ") (classification)\n");
        if (output != null)
            output.close();
    }


    private void do_cross_validation(svm_problem prob, svm_parameter param, int nr_fold) {
        int i;
        int total_correct = 0;
        double total_error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double[] target = new double[prob.l];

        svm.svm_cross_validation(prob, param, nr_fold, target);
        for (i = 0; i < prob.l; i++)
            if (target[i] == prob.y[i])
                ++total_correct;
        System.out.print("Cross Validation Accuracy = " + 100.0 * total_correct / prob.l + "%\n");
    }

}