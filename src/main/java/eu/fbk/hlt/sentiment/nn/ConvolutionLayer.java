package eu.fbk.hlt.sentiment.nn;

import eu.fbk.hlt.sentiment.nn.duyu.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A convolution layer
 * Which uses a sliding window over an unbound input to populate set of LinearTanh layers with averaging in the end
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class ConvolutionLayer implements NNInterface {
    final static Logger logger = LoggerFactory.getLogger(ConvolutionLayer.class);

    protected LinearLayer linearProt;
    protected TanhLayer tanhProt;
    protected int elementDim;
    protected int windowDim;
    protected int outputDim;
    protected List<Pipeline> layers = new ArrayList<>();
    protected NNInterface lastLayer;
    protected NNInterface nextLayer;
    protected int nextId;

    protected double[] input;
    protected double[] inputG;

    public ConvolutionLayer(int windowDim, int elementDim, int outputDim) {
        assert windowDim > 0 && elementDim > 0 && outputDim > 0;
        Random rnd = new Random();
        double rndBase = -0.01;
        linearProt = new LinearLayer(windowDim * elementDim, outputDim);
        linearProt.randomize(rnd, -1.0 * rndBase, rndBase);
        tanhProt = new TanhLayer(outputDim);
        this.elementDim = elementDim;
        this.windowDim = windowDim;
        this.outputDim = outputDim;
    }

    @Override
    public void randomize(Random r, double min, double max) {

    }

    public void setInput(double[] input) {
        this.input = input;
        this.inputG = new double[input.length];
        //The size of an input is unknown up to this point so actual layer generation occurs here
        try {
            generateLayers(input.length);
        } catch (Exception e) {
            logger.error("Can't generate layers", e);
        }
        //Fill the inputs of each underlying layer
        int offset = 0;
        for (Pipeline layer : layers) {
            double[] output = (double[]) layer.getInputLayer().getInput(0);
            System.arraycopy(input, offset, output, 0, output.length);
            offset += elementDim;
        }
    }

    @Override
    public void forward() {
        assert layers.size() > 0;
        layers.forEach(value -> value.forward());
    }

    private void generateLayers(int inputDim) throws Exception {
        assert inputDim % elementDim == 0;
        assert inputDim >= windowDim * elementDim;
        layers.clear();
        int length = inputDim / elementDim - windowDim + 1;
        int[] outputDims = new int[length];
        Arrays.fill(outputDims, outputDim);
        Pipeline connect = new Pipeline(new MultiConnectLayer(outputDims));
        for (int i = 0; i < length; i++) {
            Pipeline pipeline = new Pipeline((LinearLayer) linearProt.cloneWithTiedParams());
            pipeline.after((TanhLayer) tanhProt.cloneWithTiedParams()).after(connect, i);
            layers.add(pipeline);
        }
        lastLayer = new AverageLayer(outputDim*length, outputDim);
        connect.after(lastLayer);
        lastLayer.link(nextLayer, nextId);
    }

    @Override
    public void backward() {
        assert layers.size() > 0;
        layers.forEach(value -> value.backward());
    }

    @Override
    public void update(double learningRate) {
        assert layers.size() > 0;
        layers.forEach(value -> value.update(learningRate));
    }

    @Override
    public void updateAdaGrad(double learningRate, int batchsize) {

    }

    @Override
    public void clearGrad() {
        layers.forEach(value -> value.clearGrad());
    }

    @Override
    public void link(NNInterface nextLayer, int id) throws Exception {
        this.nextLayer = nextLayer;
        this.nextId = id;
        if (lastLayer != null) {
            lastLayer.link(nextLayer, nextId);
        }
    }

    @Override
    public void link(NNInterface nextLayer) throws Exception {
        link(nextLayer, 0);
    }

    @Override
    public Object getInput(int id) {
        return input;
    }

    @Override
    public Object getOutput(int id) {
        return lastLayer.getOutput(id);
    }

    @Override
    public Object getInputG(int id) {
        return inputG;
    }

    @Override
    public Object getOutputG(int id) {
        return lastLayer.getOutputG(id);
    }

    @Override
    public Object cloneWithTiedParams() {
        return null;
    }
}
