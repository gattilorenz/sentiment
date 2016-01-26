package eu.fbk.hlt.sentiment.util;

import org.apache.commons.cli.*;

/**
 * Write fucking description!
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class SentimentParameters {
    public static final String DEFAULT_EMBEDDINGS = "glove.6B.50d";
    public static final String DEFAULT_DATASET = "lorenzo.tweet";

    public String dataset;
    public String embeddings;
    public boolean enableStatistics;
    public boolean dumpModel;

    public String targetFolder;
    public String sentencesFilename;
    public String unknownWordsFilename;
    public String trainingStatsFilename;

    public SentimentParameters(String[] args) throws ParseException {
        //Defaults
        enableStatistics = false;
        dumpModel = false;

        targetFolder = "target";
        sentencesFilename = "sentences.tsv";
        unknownWordsFilename = "unknown_words.tsv";
        trainingStatsFilename = "training_stats.txt";

        dataset = DEFAULT_DATASET;
        embeddings = DEFAULT_EMBEDDINGS;

        //Defining input parameters
        Options options = new Options();
        CLIOptionBuilder builder = new CLIOptionBuilder().hasArg().withArgName("directory");

        options.addOption(builder.withDescription("Target directory for the results of analysis").withLongOpt("target").toOption("t"));
        options.addOption(new CLIOptionBuilder().withDescription("Dump the sentence model of the input instead of training the network").withLongOpt("dump-model").toOption("dm"));
        options.addOption(new CLIOptionBuilder().withDescription("Enable a web server with statistics (on port 8080)").withLongOpt("enable-statistics").toOption("es"));
        options.addOption(new CLIOptionBuilder().hasArg().withArgName("dataset").withDescription("Training dataset name from the repository").withLongOpt("dataset").toOption("d"));

        //Parsing the input
        CommandLineParser parser = new PosixParser();
        CommandLine line;
        try {
            //Parse the command line arguments
            line = parser.parse(options, args);

            //Filling the initial configuration
            enableStatistics = line.hasOption("enable-statistics");
            dumpModel = line.hasOption("dump-model");
            String target = line.getOptionValue("target");
            if (target != null) {
                this.targetFolder = target;
            } else {
                if (dumpModel) {
                    throw new ParseException("If you want to dump a model then target file is a required parameter");
                }
            }
            String dataset = line.getOptionValue("dataset");
            if (dataset != null) {
                this.dataset = dataset;
            }
        } catch (ParseException e) {
            //If parameters are wrong — print help
            new HelpFormatter().printHelp(400, "java -Dfile.encoding=UTF-8 <classname>", "\n", options, "\n", true);
            throw e;
        }
    }
}