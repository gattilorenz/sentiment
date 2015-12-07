package eu.fbk.hlt.sentiment.nn.duyu;

import java.util.Random;

public class SoftmaxLayer implements NNInterface{

	public int length;
	public int linkId;
	public double[] input;
	public double[] inputG;
	public double[] output;
	public double[] outputG;
	
	public SoftmaxLayer() {}

	public SoftmaxLayer(int xLength)
	{
		this(xLength, 0);
	}
	
	public SoftmaxLayer(int xLength, int xLinkId)
	{
		length = xLength;
		linkId = xLinkId;
		input = new double[length];
		inputG = new double[length];
		output = new double[length];
		outputG = new double[length];
	}
	
	@Override
	public void randomize(Random r, double min, double max) {
		
	}

	@Override
	public void forward() {
		double max = input[0];

        for (int i = 1; i < input.length; ++i)
        {
            if (input[i] > max)
            {
                max = input[i];
            }
        }

        double sum = 0;

        for (int i = 0; i < input.length; ++i)
        {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < input.length; ++i)
        {
            output[i] /= sum;
        }
	}

	@Override
	public void backward() {
		for(int i = 0; i < length; i++)
		{
			inputG[i] = 0;
		}
		
		for(int i = 0; i < length; i++)
		{
			if (outputG[i] == 0)
            {
                continue;
            }
			
			for(int j = 0; j < length; j++)
			{
				if(i == j)
				{
					inputG[j] += outputG[i] * output[i] * (1 - output[j]);
				}
				else
				{
					inputG[j] += -outputG[i] * output[j] * output[i];
				}
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
		return new SoftmaxLayer(length, linkId);
	}

}
