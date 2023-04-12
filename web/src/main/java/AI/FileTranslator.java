package AI;


import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import javax.swing.plaf.ToolTipUI;
import java.nio.file.Paths;

public class FileTranslator implements Translator<String, float[]> {

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
        return list.get(0).toFloatArray();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String input) throws Exception {
        Image image = ImageFactory.getInstance().fromFile(Paths.get(input));
        image = image.resize(112,112, true);
        NDArray input_data = image.toNDArray(ctx.getNDManager());
        input_data = input_data.toType(DataType.FLOAT32, true);
        input_data = NDImageUtils.toTensor(input_data);
        input_data.div(255);
        return new NDList(input_data);
    }
}
