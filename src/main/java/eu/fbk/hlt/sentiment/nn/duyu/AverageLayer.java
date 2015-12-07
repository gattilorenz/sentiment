package eu.fbk.hlt.sentiment.nn.duyu;

import java.util.Random;

public class AverageLayer implements NNInterface {

	public int inputLength;
    public int outputLength;

    public double[] input;
    public double[] inputG;
    
    public double[] output;
    public double[] outputG;

    public int linkId;
	
    public AverageLayer() {}

    public AverageLayer(int xInputLength, int xOutputLength)
    {
    	this(xInputLength, xOutputLength, 0);
    }
    
    public AverageLayer(int xInputLength, int xOutputLength, int xLinkId)
    {
    	inputLength = xInputLength;
    	outputLength = xOutputLength;
		linkId = xLinkId;
		input = new double[inputLength];
		inputG = new double[inputLength];
		output = new double[outputLength];
		outputG = new double[outputLength];
    }
    
	@Override
	public void randomize(Random r, double min, double max) {
	}

	@Override
	public void forward() {
		for(int i = 0; i < outputLength; i++)
		{
			output[i] = 0;
		}
		
		int K = inputLength / outputLength;
		for(int j = 0; j < K; j++)
		{
			for(int i = 0; i < outputLength; i++)
			{
				output[i] += input[j * outputLength + i];
			}
		}
		
		for(int i = 0; i < outputLength; i++)
		{
			output[i] = output[i] / K;
		}
	}

	@Override
	public void backward() {
		int K = inputLength / outputLength;
		for(int j = 0; j < K; j++)
		{
			for(int i = 0; i < outputLength; i++)
			{
				inputG[i + j * outputLength] = outputG[i] / K;
			}
		}
	}

	@Override
	public void update(double learningRate) {
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		
	}

	@Override
	public void clearGrad() {
		for(int i = 0; i < outputG.length; i++)
		{
			outputG[i] = 0;
		}
		
		for(int i = 0; i < inputG.length; i++)
		{
			inputG[i] = 0;
		}
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != output.length || nextIG.length != outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		output = nextI;
		outputG = nextIG;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		return input;
	}

	@Override
	public Object getOutput(int id) {
		return output;
	}

	@Override
	public Object getInputG(int id) {
		return inputG;
	}

	@Override
	public Object getOutputG(int id) {
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		return new AverageLayer(inputLength, outputLength, linkId);
	}

}
