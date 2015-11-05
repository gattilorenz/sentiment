package eu.fbk.hlt.sentiment;

import java.util.HashMap;

/**
 * A model for gold sentences
 *
 * @author Lorenzo Gatti (gattilorenz@gmail.com)
 */
public class GoldSentence {
	public String sentence;
	public int goldValue;
	public HashMap<String,Integer> scoreValues;
	
	public GoldSentence(String sentence, Integer goldValue) {
		this.sentence = sentence;
		this.goldValue = goldValue;
        this.scoreValues = new HashMap<>();
	}

    public void addScore(String modelName, int score) {
        scoreValues.put(modelName, score);
    }
}
