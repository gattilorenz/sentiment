package eu.fbk.hlt.sentiment.nn;

import eu.fbk.hlt.sentiment.nn.duyu.LinearLayer;
import eu.fbk.hlt.sentiment.nn.duyu.NNInterface;
import eu.fbk.hlt.sentiment.nn.duyu.TanhLayer;

import java.util.Random;

/**
 * A convolution layer
 * Which uses a sliding window over an unbound input to populate set of LinearTanh layers with averaging in the end
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class ConvolutionLayer implements NNInterface {


    public ConvolutionLayer(int windowDim, int elementDim, int outputDim) {
        LinearLayer linear = new LinearLayer(windowDim * elementDim, outputDim);
        TanhLayer tanh = new TanhLayer(outputDim);

    }

    @Override
    public void randomize(Random r, double min, double max) {

    }

    @Override
    public void forward() {

    }

    @Override
    public void backward() {

    }

    @Override
    public void update(double learningRate) {

    }

    @Override
    public void updateAdaGrad(double learningRate, int batchsize) {

    }

    @Override
    public void clearGrad() {

    }

    @Override
    public void link(NNInterface nextLayer, int id) throws Exception {

    }

    @Override
    public void link(NNInterface nextLayer) throws Exception {

    }

    @Override
    public Object getInput(int id) {
        return null;
    }

    @Override
    public Object getOutput(int id) {
        return null;
    }

    @Override
    public Object getInputG(int id) {
        return null;
    }

    @Override
    public Object getOutputG(int id) {
        return null;
    }

    @Override
    public Object cloneWithTiedParams() {
        return null;
    }
}
