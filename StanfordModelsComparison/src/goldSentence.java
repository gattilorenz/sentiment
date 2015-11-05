import java.util.HashMap;

public class goldSentence {
	public String sentence;
	public int goldValue;
	public HashMap<String,Integer> scoreValues;
	
	public goldSentence(String sentence, Integer goldValue) {
		this.sentence = sentence;
		this.goldValue = goldValue.intValue();
	}

	public goldSentence(String sentence, Integer goldValue, HashMap<String,Integer> scoreValues) {
		this.sentence = sentence;
		this.goldValue = goldValue.intValue();
		this.scoreValues = scoreValues;
	}
	
}
