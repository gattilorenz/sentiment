import java.io.*;
import java.util.*;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.*;

public class modelsComparison {

	static ArrayList<HashMap<String,String>> modelsAndFiles;

	public static void main(String[] args) throws IOException {
		for (int i = 0; i < args.length; i++) {
			System.err.println(args[i]);
		}
		System.err.println("----------\n");
		String inputFileName;
		String outputFileName;
		if (args.length>1) {
			inputFileName = args[0];
			outputFileName = args[1];
		}
		else {
			System.err.println("You need to specify as arguments:\n"+
					"1) the name of the input file\n"+
					"2) the name of the output file\n"+
					"3) the name and path of every sentiment model you want to test\n" +
					"   (e.g. [-morpho=]./model-morpho.ser.gz)\n"+
					"   (Stanford is always included)\n");
			return;
		}

		modelsAndFiles = new ArrayList<HashMap<String,String>>();
		HashMap<String,String> map = new HashMap<String,String>();
		map.put("path","");
		map.put("name","stanford");
		modelsAndFiles.add(map);

		for (int i = 2; i < args.length; i++) {
			String argument = args[i].replaceAll("^-+", "");
			String[] arg = argument.split("=");
			String modelName;
			String modelPath;
			if (arg.length>2) {
				System.err.println("The syntax for each model is:");
				System.err.println("[-modelname=]modelpath");
				return;
			}
			else {
				if (arg.length==2) {
					modelName=arg[0];
					modelPath=arg[1];					
				}
				else {
					modelPath=arg[0];
					modelName=arg[0];
					modelName.replaceAll("^.*/", "");
					modelName.replaceAll("\\.ser\\.gz", "");
				}
			}

			HashMap<String,String> tmpMap = new HashMap<String,String>();
			tmpMap.put("path",modelPath);
			tmpMap.put("name",modelName);
			modelsAndFiles.add(tmpMap);
		}


		long startTime = System.currentTimeMillis();
		ArrayList<goldSentence> entries = readFile(inputFileName);
		entries = annotateScores(entries, modelsAndFiles);
		long estimatedTime = System.currentTimeMillis() - startTime;
		System.err.println("Reading and annotating the file took about " + (estimatedTime/1000)+" seconds");
		saveResults(entries,outputFileName);

	}


	public static ArrayList<goldSentence> readFile (String fileName) {		
		ArrayList<goldSentence> result = new ArrayList<goldSentence>();
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			String line;
			while ((line = br.readLine()) != null) {

				String[] tokenizedLine = line.split("\t");
				if (tokenizedLine.length==2) {
					int goldValue = Integer.parseInt(tokenizedLine[0]);
					String sentence = tokenizedLine[1];
					result.add(new goldSentence(sentence,goldValue));
				}
				else {
					System.err.println("WARNING: line does not have two tabs\n("+line+")");
				}
			}
		}
		catch (Exception e) {
			System.err.println(e.toString());
		}
		return result;
	}

	private static ArrayList<goldSentence> annotateScores (ArrayList<goldSentence> entries, ArrayList<HashMap<String,String>> sentimentModelsAndPaths) {
		ArrayList<goldSentence> returnValue = new ArrayList<goldSentence>();
		//incremental annotation: tokenize, split and parse ONCE
		//...THEN add sentiment
		Properties commonProps = new Properties();
		commonProps.setProperty("annotators", "tokenize, ssplit, parse");
		StanfordCoreNLP commonPipeline = new StanfordCoreNLP(commonProps);
		BinarizerAnnotator  binarizerAnnotator = new BinarizerAnnotator("ba", new Properties());
		commonPipeline.addAnnotator(binarizerAnnotator);

		//now put all the sentences in an array of Annotation
		ArrayList<Annotation> sentencesToAnnotate = new ArrayList<Annotation>();
		for (goldSentence entry : entries) {
			sentencesToAnnotate.add(new Annotation(entry.sentence));
		}
		//...and annotate everything in parallel
		commonPipeline.annotate(sentencesToAnnotate);

		//initialize the different sentiment models
		ArrayList<SentimentAnnotator> sentimentModels = new ArrayList<SentimentAnnotator>();
		for (HashMap<String,String> model : modelsAndFiles) {
			Properties props = new Properties();
			if (!model.get("name").equals("stanford") && model.get("path").length()>0) {
				props.setProperty("sentiment.model",model.get("path"));
			}
			SentimentAnnotator sentimentAnnotator = new SentimentAnnotator("sentiment",props);
			sentimentModels.add(sentimentAnnotator);
		}

		//for (Annotation annotatedSentence : sentencesToAnnotate) {
		for (int j = 0; j < entries.size(); j++) {
			Annotation annotatedSentence = sentencesToAnnotate.get(j);
			HashMap<String, Integer> annotatedValues = new HashMap<String,Integer>();
			for (int i=0; i < modelsAndFiles.size(); i++) {
				String nameOfModel = modelsAndFiles.get(i).get("name");
				SentimentAnnotator sentimentAnnotator = sentimentModels.get(i);
				Integer sentenceValue = annotateSentiment(annotatedSentence, sentimentAnnotator);
				annotatedValues.put(nameOfModel,sentenceValue);
			}
			returnValue.add(new goldSentence(entries.get(j).sentence,entries.get(j).goldValue,annotatedValues));
		}

		return returnValue;
	}

	private static Integer annotateSentiment(Annotation annotation, SentimentAnnotator sentimentAnnotator) {
		int mainSentiment = 0;
		int longest = 0;
		//add sentiment annotation
		Annotation tmpAnnotation = annotation.copy();
		sentimentAnnotator.annotate(tmpAnnotation);
		for (CoreMap sent : tmpAnnotation.get(CoreAnnotations.SentencesAnnotation.class)) {
			Tree tree = sent.get(SentimentCoreAnnotations.AnnotatedTree.class);
			int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
			String partText = sent.toString();
			if (partText.length() > longest) {
				mainSentiment = sentiment;
				longest = partText.length();
			}
		}
		return new Integer(mainSentiment);
	}

	private static void saveResults(ArrayList<goldSentence> entries, String fileName) {
		if (entries.size()<1) {
			return;
		}
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter(new FileWriter(fileName));
			ArrayList<String> models = new ArrayList<String>();
			bw.write("SENTENCE\tGOLD");
			for (String model : entries.get(0).scoreValues.keySet()) {
				models.add(model);
				bw.write("\t"+model);
			}
			bw.write("\n");
			for (goldSentence entry : entries) {
				String lineToWrite = entry.sentence+"\t"+entry.goldValue;
				for (String model : models) {
					lineToWrite = lineToWrite +"\t"+entry.scoreValues.get(model); 
				}
				bw.write(lineToWrite+"\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		finally {
			try {
				bw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

}



