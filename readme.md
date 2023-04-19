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

## Usage

To train your own network, you might first run `traning_data_generation.py` to generate the training data
```
python traning_data_generation.py
```

Then, run train.py to train the network
```
python train.py
```

After training, run `generate_data_base.py` to generate the database for query

```
python generate_database.py
```

At last, run `test_database.py` to scognize the file `test.png`

```
python test_database.py
```

note that if you want to evalueate the accuracy on a certain font, you may modify `evaluate.py`. This file recognize a character via a web API (we establish it on a local server). The code for the server is in `web` directory. You should first put the database into the resource directory since it's a little big that we did not upload to the repository.
