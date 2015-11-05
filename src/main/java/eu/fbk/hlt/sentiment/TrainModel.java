package eu.fbk.hlt.sentiment;

import eu.fbk.hlt.sentiment.util.CLIOptionBuilder;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.util.List;
import java.util.Properties;

/**
 * Gets the dataset and trains the model using Stanford NLP
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class TrainModel {
    final static Logger logger = LoggerFactory.getLogger(TrainModel.class);

    public static void main(String[] args) {
        Options options = new Options();
        CLIOptionBuilder builder = new CLIOptionBuilder().hasArg().withArgName("file");

        options.addOption(builder.withDescription("Target file with model").withLongOpt("target").toOption("s"));
        options.addOption(builder.withDescription("Source dataset").withLongOpt("source").toOption("source"));

        CommandLineParser parser = new PosixParser();
        CommandLine line;
        try {
            //Parse the command line arguments
            line = parser.parse(options, args);

            //Bootstrapping configuration
            Properties properties = new Properties();
            String propertyFile = line.getOptionValue("target");
            if (propertyFile != null) {
                properties.load(new FileInputStream(propertyFile));
            }


        } catch (Exception e) {
            logger.error(e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
