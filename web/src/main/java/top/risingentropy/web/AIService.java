package top.risingentropy.web;

import AI.AIEngine;
import AI.Result;
import ai.djl.translate.TranslateException;
import jakarta.annotation.PostConstruct;
import org.apache.juli.logging.Log;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

@RestController
public class AIService {
    @Value("${AI.model_path}")
    private String modelPath;
    @Value("${AI.database}")
    private String databasePath;
    @Value("${AI.threshold:0.85f}")
    private float confThreshold;
    private AIEngine engine;
    private Logger logger = LoggerFactory.getLogger(AIService.class);
    @PostConstruct
    public void init() throws IOException {
        this.engine = new AIEngine();
        engine.initialize(modelPath);
        engine.loadDatabase(databasePath);
        if(!engine.isInitialized()){
            logger.error("AI model initialization failed!");
        }
        if(!engine.isDatabaseLoaded()){
            logger.error("AI database not loaded!");
        }
        if(engine.engineOK()){
            logger.info("AI engine ready");
        }
    }
    @RequestMapping(value = "/ai/detect", method = RequestMethod.POST)
    public String handleRequest(@RequestParam("img")String imgBase64,
                                @RequestParam(value = "norm_threshold", required = false, defaultValue = "0.85")float confThreshold){
        logger.trace("New Request established");
        logger.info("Start to infer");
        Result res = this.engine.infer(imgBase64, confThreshold);
        logger.info("Inferring complete with threshold "+ confThreshold);
        if(!res.hasValidData()){
            logger.info("A wrong inferring occurred");
        }
        return res.toJson();
    }
    @RequestMapping(value="/ai/test", method = RequestMethod.POST)
    public String AITest(@RequestParam("img")String imgBase64){
        return "echo:"+imgBase64;
    }
}
