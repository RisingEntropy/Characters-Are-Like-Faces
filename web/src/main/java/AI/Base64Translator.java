package AI;

import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.translate.Transform;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;
import java.util.Base64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Base64Translator implements Translator<String,float[]> {
    private static final String BASE64_PREFIX = "data:image/jpeg;base64,";
    private final Resize resizer;
    private final ToTensor toTensor;
    private Logger logger = LoggerFactory.getLogger(this.getClass());
    public Base64Translator(){
        this.resizer = new Resize(112,112);
        this.toTensor = new ToTensor();
    }
    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return list.get(0).toFloatArray();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) throws Exception {
        if(input.startsWith(BASE64_PREFIX)){
            input = input.substring(BASE64_PREFIX.length());
        }
        ByteArrayInputStream inputStream = new ByteArrayInputStream(Base64.getDecoder().decode(input));
        Image image = BufferedImageFactory.getInstance().fromInputStream(inputStream);
        NDArray tensor = image.toNDArray(ctx.getNDManager());
        tensor = this.resizer.transform(tensor);
        tensor = this.toTensor.transform(tensor);
        tensor.div(255);
        return new NDList(tensor);
    }
}
