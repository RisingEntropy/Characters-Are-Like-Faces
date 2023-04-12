# Characters Are Like Faces
This is the source code for ICLR 2023 TinyPaper *Characters Are Like Faces*
## File explanation
**character_argumentation.py**: part of argumentation approach mentioned in the paper

**dataset.p**y: the dataset class used to load training data

**evaluate.py**: evaluation through our website(we didn't utilize python with pytorch to evaluate simply because it's too slow. We established a local http server to do this, code can be found in the web directory)

**export_model.py**: to export our trained model to onnx format, making it convenience to deploy on our web server

**generate_database.py**: generate the final database using font files in `eval_fonts` directory with trained model.

**modules.py**: the definition of our network, including MobileFaceNetV3 and Arcface Loss

**test_database.py**: reconginze the `test.png` file and give out the possible result using python with pytorch, it's super slow!

**train.py**: training script

**training_data_generation.py**: generate our training data using font files in the fonts directory
