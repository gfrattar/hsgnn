In order to use your trained model in Athena you need to export it to [ONNX](https://onnxruntime.ai/).


### Model Export

The `to_onnx.py` python script handles the ONNX conversion process for you.
The script has several arguments, you can learn about them by running

```bash
to_onnx --help
```

At a minimum, you need to specify the path to a checkpoint to convert, and a track selection.
For example

```bash
to_onnx \
    --ckpt_path logs/<timestamp>/ckpts/checkpoint.ckpt \
    --track_selection r22default
```

If you don't specify a config path using `--config`, the script will look for one in the parent of the `--ckpt_path`.

??? warning "Track selection"

    The track selection you specify must correspond to one of the options defined in `trk_select_regexes` variable in
    [`DataPrepUtilities.cxx`](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/Root/DataPrepUtilities.cxx).

    The selection you use must also match the selection applied in your training samples.
    Track selection is applied when dumping using the TDD.
    The current default FTAG selection is called `r22default`, but you should take note of the changes described in
    [training-dataset-dumper!427](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/merge_requests/427)
    to make sure you are using the correct selection.

You can also optionally specify a different scale dict to the one in the training config, and a model name (by default this is `salt`).
The model name is used to construct the output probability variable names in Athena.


### Athena Validation

You may see some warnings during export, but the `to_onnx` script will verify that there is a good level of compatability between the pytorch and ONNX model outputs, and that there are no `nan` or `0` values in the output.
However, as a final check, you should verify the performance of your pytorch model against a version running from the TDD.

First, follow the instructions [here](https://training-dataset-dumper.docs.cern.ch/configuration/#dl2-config) to dump the scores of your export model.
Please take note of the following considerations when comparing Athena and Python evaluated models:

- Models in Athena are evaluated with full precision inputs. Make sure to dump using the TDD at full precision (use the provided flag `--force-full-precision`).
- Models evaluated in Python are limited to 40 input tracks, whereas models evaluated in Athena have no such limit.


Once you have evaluated your model using the TDD, you should use the resulting h5 file to run `salt test`.
Be sure to run with `--trainer.precision 32`.

Finally, you can then the `compare_models` command to compare the scores of the two models.

```bash
compare_models \
    --file_A tdd/output.h5 \
    --tagger_A name \
    --file_B salt/eval.h5 \
    --tagger_B name
```

See `compare_models.py -h` for more information.

??? info "What level of discrepancy is expected?"

    We usually ask for agreement within `1e-6` for the output probabilities, which is approximately floating point precision error.
    If you see one or two jets with a discrepancy of `1e-5`, this is probably fine.
    Common causes of more significant discrepancies are:

    - Not dumping at full precision using the TDD (see above)
    - Not running `salt test` with `--trainer.precision 32` if you trained at lower precision.
    - Not writing out salt evaluation scores at full precision (see the `PredictionWriter` callback)
    - Enabling some runtime optimisaiton in pytorch (e.g. [here](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision))


### Viewing ONNX Model Metadata

To view the metadata stored in an ONNX file, you can use

```bash
get_onnx_metadata path/to/model.onnx
```

Inside are the list of input features including normalisation values, and also the list of outputs and the model name.

??? info "A command with the same name is also available in Athena"

    After setting up Athena, you can also run a different [`get_onnx_metadata`](https://gitlab.cern.ch/atlas/athena/-/blob/master/PhysicsAnalysis/JetTagging/FlavorTagDiscriminants/util/get-onnx-metadata.cxx) command which has the same function.
