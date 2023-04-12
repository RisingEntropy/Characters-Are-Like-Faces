package AI;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Objects;

import ai.djl.util.Pair;
import com.alibaba.fastjson2.*;
public class Result implements Serializable {
    private String bestMatch = null;
    private float norm = 1e9f;

    private final ArrayList<String> candidates;
    private final ArrayList<Float> candidate_norms;
    public Result(){
        this.candidates = new ArrayList<>();
        this.candidate_norms = new ArrayList<>();
    }
    public boolean hasValidData(){
        return this.bestMatch!=null;
    }
    public void setBestMatch(String chr, float norm){
        this.bestMatch = chr;
        this.norm = norm;
    }
    public void addCandidate(String chr, float norm){
        this.candidates.add(chr);
        this.candidate_norms.add(norm);
    }
    public String getBestMatch(){
        return this.bestMatch;
    }
    public float getBestMatchNorm(){
        return this.norm;
    }
    public String getCandidateByIndex(int index){
        if(index>=this.candidates.size()){
            return "Index Error!!!";
        }
        return this.candidates.get(index);
    }
    public float getCandidateNormByIndex(int index){
        if(index>=this.candidate_norms.size())
            return 1e9f;
        return this.candidate_norms.get(index);
    }
    @Override
    public String toString(){
        StringBuffer buffer = new StringBuffer();
        buffer.append("chr:").append(this.bestMatch).append(" norm:").append(Float.toString(this.norm)).append("\n");
        for(int i = 0;i<this.candidates.size();i++){
            buffer.append("candidate chr:"+candidates.get(i)+" norm:"+ candidate_norms.get(i)+"\n");
        }
        return buffer.toString();
    }

    public String toJson(){
        JSONObject json = new JSONObject();
        if(this.bestMatch==null){
            json.put("status","error");
            return json.toJSONString();
        }
        json.put("status","OK");
        json.put("bestMatch",this.bestMatch);
        json.put("norm",String.format("%.2f", this.norm));
        JSONArray candidate = new JSONArray();
        ArrayList<Pair<String,Float> > list = new ArrayList<>();
        for(int i = 0;i<this.candidates.size();i++){
            list.add(new Pair<>(candidates.get(i),candidate_norms.get(i)));
        }
        list.sort((a, b) -> Objects.equals(a.getValue(), b.getValue()) ? 0 : a.getValue().compareTo(b.getValue()));
        for(int i = 0;i<list.size();i++){
            JSONObject candi = new JSONObject();
            candi.put("character",list.get(i).getKey());
            candi.put("norm",String.format("%.2f", list.get(i).getValue()));
            candidate.add(candi);
        }
        json.put("candidates",candidate);
        return json.toJSONString();
    }
}
