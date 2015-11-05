# Sentiment analysis tool

## Usage

Compiling and packaging:

`mvn package`

Models comparison script:

`java -cp sentiment-0.1.jar -Xmx2g eu.fbk.hlt.sentiment.ModelsComparison [-M <modelname=modelpath>] -s <file> -t <file>`