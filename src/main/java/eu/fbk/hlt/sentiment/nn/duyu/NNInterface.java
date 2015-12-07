package eu.fbk.hlt.sentiment.nn.duyu;

import java.util.Random;

public interface NNInterface {
	void randomize(Random r, double min, double max);
	
	void forward();
	
	void backward();
	
	void update(double learningRate);
	
	void updateAdaGrad(double learningRate, int batchsize);
	
	void clearGrad();
	
	void link(NNInterface nextLayer, int id) throws Exception;

    void link(NNInterface nextLayer) throws Exception;

    Object getInput(int id);

    Object getOutput(int id);

    Object getInputG(int id);

    Object getOutputG(int id);
    
    Object cloneWithTiedParams();
}
