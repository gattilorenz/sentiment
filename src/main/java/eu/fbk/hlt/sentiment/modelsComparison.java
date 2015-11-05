package eu.fbk.hlt.sentiment;

import java.io.*;
import java.util.*;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.*;
import eu.fbk.hlt.sentiment.util.CLIOptionBuilder;
import eu.fbk.hlt.sentiment.util.Stopwatch;
import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Compares the results of different models over the same dataset
 *
 * @author Lorenzo Gatti (gattilorenz@gmail.com), Yaroslav Nechaev (remper@me.com)
 */
public class ModelsComparison {
    final static Logger logger = LoggerFactory.getLogger(ModelsComparison.class);

	ArrayList<Model> modelsAndFiles;
    String inputFileName;
    String outputFileName;

    public ModelsComparison(String inputFileName, String outputFileName) {
        this.inputFileName = inputFileName;
        this.outputFileName = outputFileName;
        this.modelsAndFiles = new ArrayList<>();
        //Stanford is always included
        this.addModel("stanford", "");
    }

    public void addModel(String name, String path) {
        Model model = new Model();
        model.name = name;
        model.path = path;
        modelsAndFiles.add(model);
    }

    /**
     * Main workflow
     */
    public void start() {
        logger.info("Loaded " +modelsAndFiles.size()+ " models:");
        for (Model model : modelsAndFiles) {
            logger.info("  "+model.name+" "+model.path);
        }
        Stopwatch watch = Stopwatch.start();
        ArrayList<GoldSentence> entries = readFile(inputFileName);
        annotateScores(entries);
        logger.info("Reading and annotating the file took about " + (watch.click()/1000) + " seconds");
        saveResults(entries, outputFileName);
    }

    /****************************************
     * An entry point and parameter parsing *
     ****************************************/
    final static String BANNER_STRING =
            "You need to specify as arguments:\n" +
            "1) the path to the input file\n" +
            "2) the path to the output file\n" +
            "3) the name and path of every sentiment model you want to test (Stanford is always included)\n";

	public static void main(String[] args) throws IOException {
        //Defining input parameters
        Options options = new Options();
        CLIOptionBuilder builder = new CLIOptionBuilder().hasArg().withArgName("file").isRequired();

        options.addOption(builder.withDescription("Target file with model").withLongOpt("target").toOption("t"));
        options.addOption(builder.withDescription("Source dataset").withLongOpt("source").toOption("s"));

        Option modelOption = new CLIOptionBuilder().hasArgs(2)
                .withArgName("modelname=modelpath")
                .withValueSeparator()
                .withDescription("a set of sentiment models you want to test")
                .withLongOpt("models")
                .toOption("M");
        options.addOption(modelOption);

        //Parsing the input
        CommandLineParser parser = new PosixParser();
        CommandLine line;
        ModelsComparison modelsComparison;
        try {
            //Parse the command line arguments
            line = parser.parse(options, args);

            //Filling the initial configuration
            String inputFileName = line.getOptionValue("source");
            String outputFileName = line.getOptionValue("target");
            modelsComparison = new ModelsComparison(inputFileName, outputFileName);

            //Parsing the models
            Properties models = line.getOptionProperties("models");
            for (Map.Entry model : models.entrySet()) {
                String modelName = (String) model.getKey();
                String modelPath = (String) model.getValue();
                modelsComparison.addModel(modelName, modelPath);
            }
        } catch (ParseException e) {
            //If parameters are wrong — print help
            new HelpFormatter().printHelp(400, "java -Dfile.encoding=UTF-8 " + ModelsComparison.class.getName(), BANNER_STRING, options, "\n", true);
            return;
        }

        try {
            modelsComparison.start();
        } catch (Exception e) {
            //Print out any nasty exceptions
            logger.error(e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
	}

	public static ArrayList<GoldSentence> readFile (String fileName) {
		ArrayList<GoldSentence> result = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			String line;
			while ((line = br.readLine()) != null) {
				String[] tokenizedLine = line.split("\t");

				if (tokenizedLine.length == 2) {
					int goldValue = Integer.parseInt(tokenizedLine[0]);
					String sentence = tokenizedLine[1];
					result.add(new GoldSentence(sentence,goldValue));
				} else {
					logger.error("WARNING: line does not have two tabs\n(" + line + ")");
				}
			}
		} catch (Exception e) {
			logger.error(e.getClass().getSimpleName() + " " + e.getMessage());
		}
		return result;
	}

	private void annotateScores (ArrayList<GoldSentence> entries) {
		//incremental annotation: tokenize, split and parse ONCE
		//...THEN add sentiment
		Properties commonProps = new Properties();
		commonProps.setProperty("annotators", "tokenize, ssplit, parse");
		StanfordCoreNLP commonPipeline = new StanfordCoreNLP(commonProps);
		BinarizerAnnotator  binarizerAnnotator = new BinarizerAnnotator("ba", new Properties());
		commonPipeline.addAnnotator(binarizerAnnotator);

		//now put all the sentences in an array of Annotation
		ArrayList<Annotation> sentencesToAnnotate = new ArrayList<>();
		for (GoldSentence entry : entries) {
			sentencesToAnnotate.add(new Annotation(entry.sentence));
		}
		//...and annotate everything in parallel
		commonPipeline.annotate(sentencesToAnnotate);

		//initialize the different sentiment models
		HashMap<String, SentimentAnnotator> sentimentModels = new HashMap<>();
		for (Model model : modelsAndFiles) {
			Properties props = new Properties();
			if (!model.name.equals("stanford") && model.path.length()>0) {
				props.setProperty("sentiment.model" , model.path);
			}
			SentimentAnnotator sentimentAnnotator = new SentimentAnnotator("sentiment",props);
			sentimentModels.put(model.name, sentimentAnnotator);
		}

        //annotate sentiment for each model and for each sentence, write results
		for (int j = 0; j < entries.size(); j++) {
			Annotation annotatedSentence = sentencesToAnnotate.get(j);
            GoldSentence entry = entries.get(j);

            for (Map.Entry<String, SentimentAnnotator> model : sentimentModels.entrySet()) {
                String name = model.getKey();
                SentimentAnnotator annotator = model.getValue();

                entry.addScore(name, annotateSentiment(annotatedSentence, annotator));
            }
		}
	}

	private static Integer annotateSentiment(Annotation annotation, SentimentAnnotator sentimentAnnotator) {
		int mainSentiment = 0;
		int longest = 0;
		//add sentiment annotation
		Annotation tmpAnnotation = annotation.copy();
		sentimentAnnotator.annotate(tmpAnnotation);
		for (CoreMap sent : tmpAnnotation.get(CoreAnnotations.SentencesAnnotation.class)) {
			Tree tree = sent.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
			int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
			String partText = sent.toString();
			if (partText.length() > longest) {
				mainSentiment = sentiment;
				longest = partText.length();
			}
		}
		return mainSentiment;
	}

	private static void saveResults(ArrayList<GoldSentence> entries, String fileName) {
		if (entries.size()<1) {
			return;
		}
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(fileName));
			ArrayList<String> models = new ArrayList<>();
			bw.write("SENTENCE\tGOLD");
			for (String model : entries.get(0).scoreValues.keySet()) {
				models.add(model);
				bw.write("\t"+model);
			}
			bw.write("\n");
			for (GoldSentence entry : entries) {
				String lineToWrite = entry.sentence+"\t"+entry.goldValue;
				for (String model : models) {
					lineToWrite = lineToWrite +"\t"+entry.scoreValues.get(model); 
				}
				bw.write(lineToWrite+"\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if (bw != null)
                    bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

    private static class Model {
        String path;
        String name;
    }
}



