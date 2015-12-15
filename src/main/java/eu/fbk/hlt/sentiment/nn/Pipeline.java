package eu.fbk.hlt.sentiment.nn;

import eu.fbk.hlt.sentiment.nn.duyu.NNInterface;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 * A simple pipeline that memorizes the order in which forward/backward propagation has to be applied
 * Temporary class that works with Tang's code until proper implementation is introduced
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class Pipeline {
    protected ArrayList<Pipeline> pipelines = new ArrayList<>();
    protected NNInterface layer;
    protected boolean updated = false;

    public Pipeline(NNInterface layer) {
        this.layer = layer;
    }

    /**
     * Do full forward propagation pass through the structure
     */
    public void forward() {
        layer.forward();
        pipelines.forEach(value -> value.forward());
    }

    /**
     * Do full back propagation pass through the structure (calculate gradients)
     */
    public void backward() {
        if (updated) {
            return;
        }
        pipelines.forEach(value -> value.backward());
        layer.backward();
        updated = true;
    }

    /**
     * Update weights according to the precomputed gradient
     */
    public void update(double learningRate) {
        layer.update(learningRate);
        updated = false;
        pipelines.forEach(value -> value.update(learningRate));
    }

    /**
     * Clear gradient in preparation for the next iteration
     */
    public void clearGrad() {
        layer.clearGrad();
        updated = false;
        pipelines.forEach(value -> value.clearGrad());
    }

    /**
     * Calculate gradients, update weights, clear gradients
     */
    public void fullBackward(double learningRate) {
        backward();
        update(learningRate);
        clearGrad();
    }

    public Pipeline after(Pipeline pipeline) throws Exception {
        layer.link(pipeline.getInputLayer());
        pipelines.add(pipeline);
        return pipeline;
    }

    public Pipeline after(Pipeline pipeline, int socket) throws Exception {
        layer.link(pipeline.getInputLayer(), socket);
        pipelines.add(pipeline);
        return pipeline;
    }

    public Pipeline after(NNInterface layer) throws Exception {
        this.layer.link(layer);
        Pipeline next = new Pipeline(layer);
        pipelines.add(next);
        return next;
    }

    public Pipeline after(NNInterface layer, int socket) throws Exception {
        this.layer.link(layer, socket);
        Pipeline next = new Pipeline(layer);
        pipelines.add(next);
        return next;
    }

    public Pipeline link(Pipeline pipeline) throws Exception {
        layer.link(pipeline.getInputLayer());
        pipelines.add(pipeline);
        return this;
    }

    public Pipeline link(Pipeline pipeline, int socket) throws Exception {
        layer.link(pipeline.getInputLayer(), socket);
        pipelines.add(pipeline);
        return this;
    }

    public Pipeline link(NNInterface layer) throws Exception {
        this.layer.link(layer);
        pipelines.add(new Pipeline(layer));
        return this;
    }

    public Pipeline link(NNInterface layer, int socket) throws Exception {
        this.layer.link(layer, socket);
        pipelines.add(new Pipeline(layer));
        return this;
    }

    public Set<NNInterface> getOutputLayers() {
        HashSet<NNInterface> aggregator = new HashSet<>();
        if (pipelines.isEmpty()) {
            aggregator.add(layer);
            return aggregator;
        }
        for (Pipeline pipeline : pipelines) {
            aggregator.addAll(pipeline.getOutputLayers());
        }
        return aggregator;
    }

    public NNInterface getInputLayer() {
        return layer;
    }
}
