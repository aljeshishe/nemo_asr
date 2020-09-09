'''
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 torchaudio==0.5 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install wget && apt-get install sox libsndfile1 ffmpeg && pip install nemo_toolkit[asr]==0.10.0b10 && \
    pip install unidecode && mkdir configs && \
    wget -P configs/ https://raw.githubusercontent.com/NVIDIA/NeMo/master/examples/asr/notebooks/configs/jasper_an4.yaml
'''

# This is where the an4/ directory will be placed.
# Change this if you don't want the data to be extracted in the current directory.

from nemo.utils.lr_policies import CosineAnnealing

from nemo1.nemo.utils.lr_policies import get_all_lr_classes

data_dir = '.'

import glob
import os
import subprocess
import tarfile
import wget

# Download the dataset. This will take a few moments...
print("******")
if not os.path.exists(data_dir + '/an4_sphere.tar.gz'):
    an4_url = 'http://www.speech.cs.cmu.edu/databases/an4/an4_sphere.tar.gz'
    an4_path = wget.download(an4_url, data_dir)
    print(f"Dataset downloaded at: {an4_path}")
else:
    print("Tarfile already exists.")
    an4_path = data_dir + '/an4_sphere.tar.gz'

if not os.path.exists(data_dir + '/an4/'):
    # Untar and convert .sph to .wav (using sox)
    tar = tarfile.open(an4_path)
    tar.extractall(path=data_dir)

    print("Converting .sph to .wav...")
    sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
    for sph_path in sph_list:
        wav_path = sph_path[:-4] + '.wav'
        cmd = ["sox", sph_path, wav_path]
        subprocess.run(cmd)
print("Finished conversion.\n******")

"""You should now have a folder called `an4` that contains `etc/an4_train.transcription`, `etc/an4_test.transcription`, audio files in `wav/an4_clstk` and `wav/an4test_clstk`, along with some other files we will not need.

Now we can load and take a look at the data. As an example, file `cen2-mgah-b.wav` is a 2.6 second-long audio recording of a man saying the letters "G L E N N" one-by-one (feel free to check this out by listening to `./an4/wav/an4_clstk/mgah/cen2-mgah-b.wav`). In an ASR task, the WAV file would be our input, and "G L E N N" would be our desired output.

Let's plot the waveform, which is simply a line plot of the sequence of values that we read from the file. This is a format of viewing audio that you are likely to be familiar with seeing in many audio editors and visualizers:
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Plot our example audio file's waveform
# example_file = data_dir + '/an4/wav/an4_clstk/mgah/cen2-mgah-b.wav'
# audio, sample_rate = librosa.load(example_file)
#
# plt.rcParams['figure.figsize'] = (15,7)
# plt.title('Waveform of Audio Example')
# plt.ylabel('Amplitude')
#
# _ = librosa.display.waveplot(audio)

"""We can see the activity in the waveform that corresponds to each letter in the audio, as our speaker here enunciates quite clearly!
You can kind of tell that each spoken letter has a different "shape," and it's interesting to note that last two blobs look relatively similar, which is expected because they are both the letter "N."

### Spectrograms and Mel Spectrograms

However, since audio information is more useful in the context of frequencies of sound over time, we can get a better representation than this raw sequence of 57,330 values.
We can apply a [Fourier Transform](https://en.wikipedia.org/wiki/Fourier_transform) on our audio signal to get something more useful: a **spectrogram**, which is a representation of the energy levels (i.e. amplitude, or "loudness") of each frequency (i.e. pitch) of the signal over the duration of the file.
A spectrogram (which can be viewed as a heat map) is a good way of seeing how the *strengths of various frequencies in the audio vary over time*, and is obtained by breaking up the signal into smaller, usually overlapping chunks and performing a Short-Time Fourier Transform (STFT) on each.

Let's examine what the spectrogram of our sample looks like.
"""

# # Get spectrogram using Librosa's Short-Time Fourier Transform (stft)
# spec = np.abs(librosa.stft(audio))
# spec_db = librosa.amplitude_to_db(spec, ref=np.max)  # Decibels
#
# # Use log scale to view frequencies
# librosa.display.specshow(spec_db, y_axis='log', x_axis='time')
# plt.colorbar()
# plt.title('Audio Spectrogram');

"""Again, we are able to see each letter being pronounced, and that the last two blobs that correspond to the "N"s are pretty similar-looking. But how do we interpret these shapes and colors? Just as in the waveform plot before, we see time passing on the x-axis (all 2.6s of audio). But now, the y-axis represents different frequencies (on a log scale), and *the color on the plot shows the strength of a frequency at a particular point in time*.

We're still not done yet, as we can make one more potentially useful tweak: using the **Mel Spectrogram** instead of the normal spectrogram. This is simply a change in the frequency scale that we use from linear (or logarithmic) to the mel scale, which is "a perceptual scale of pitches judged by listeners to be equal in distance from one another" (from [Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)).

In other words, it's a transformation of the frequencies to be more aligned to what humans perceive; a change of +1000Hz from 2000Hz->3000Hz sounds like a larger difference to us than 9000Hz->10000Hz does, so the mel scale normalizes this such that equal distances sound like equal differences to the human ear. Intuitively, we use the mel spectrogram because in this case we are processing and transcribing human speech, such that transforming the scale to better match what we hear is a useful procedure.
"""

# # Plot the mel spectrogram of our sample
# mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate)
# mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#
# librosa.display.specshow(
#     mel_spec_db, x_axis='time', y_axis='mel')
# plt.colorbar()
# plt.title('Mel Spectrogram');

"""## Building a Simple ASR Pipeline in NeMo

Now that we have an idea of what the audio data looks like, we can start building our end-to-end ASR pipeline!

We'll be using the **Neural Modules (NeMo) toolkit** for this part, so if you haven't already, you should download and install NeMo and its dependencies. To do so, just follow the directions on the [GitHub page](https://github.com/NVIDIA/NeMo), or in the [documentation](https://nvidia.github.io/NeMo/).

NeMo lets us easily hook together the components (modules) of our model, such as the data layer, intermediate layers, and various losses, without worrying too much about implementation details of individual parts or connections between modules. If you're curious, you can read more about NeMo and how it works in the documentation pages linked above.
"""

# NeMo's "core" package
import nemo
# NeMo's ASR collection
import nemo.collections.asr as nemo_asr

"""### Creating Data Manifests

The first thing we need to do now is to create manifests for our training and evaluation data, which will contain the metadata of our audio files. NeMo data layers take in a standardized manifest format where each line corresponds to one sample of audio, such that the number of lines in a manifest is equal to the number of samples that are represented by that manifest. A line must contain the path to an audio file, the corresponding transcript (or path to a transcript file), and the duration of the audio sample.

Here's an example of what one line in a NeMo-compatible manifest might look like:
```
{"audio_filepath": "path/to/audio.wav", "duration": 3.45, "text": "this is a nemo tutorial"}
```

We can build our training and evaluation manifests using `an4/etc/an4_train.transcription` and `an4/etc/an4_test.transcription`, which have lines containing transcripts and their corresponding audio file IDs:
```
...
<s> P I T T S B U R G H </s> (cen5-fash-b)
<s> TWO SIX EIGHT FOUR FOUR ONE EIGHT </s> (cen7-fash-b)
...
```
"""

# --- Building Manifest Files --- #
import json
import librosa
# Function to build a manifest
def build_manifest(transcripts_path, manifest_path, wav_path):
    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for line in fin:
                # Lines look like this:
                # <s> transcript </s> (fileID)
                transcript = line[: line.find('(')-1].lower()
                transcript = transcript.replace('<s>', '').replace('</s>', '')
                transcript = transcript.strip()

                file_id = line[line.find('(')+1 : -2]  # e.g. "cen4-fash-b"
                audio_path = os.path.join(
                    data_dir, wav_path,
                    file_id[file_id.find('-')+1 : file_id.rfind('-')],
                    file_id + '.wav')

                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout)
                fout.write('\n')
                
# Building Manifests
print("******")
train_transcripts = data_dir + '/an4/etc/an4_train.transcription'
train_manifest = data_dir + '/an4/train_manifest.json'
if not os.path.isfile(train_manifest):
    build_manifest(train_transcripts, train_manifest, 'an4/wav/an4_clstk')
    print("Training manifest created.")

test_transcripts = data_dir + '/an4/etc/an4_test.transcription'
test_manifest = data_dir + '/an4/test_manifest.json'
if not os.path.isfile(test_manifest):
    build_manifest(test_transcripts, test_manifest, 'an4/wav/an4test_clstk')
    print("Test manifest created.")
print("******")

"""### Building Training and Evaluation DAGs

Let's take a look at the model that we will be building, and how we specify its parameters.

#### The Jasper Model

We will be putting together a small [Jasper (Just Another SPeech Recognizer) model](https://arxiv.org/abs/1904.03288).
In brief, Jasper architectures consist of a repeated block structure that utilizes 1D convolutions.
In a Jasper_KxR model, `R` sub-blocks (consisting of a 1D convolution, batch norm, ReLU, and dropout) are grouped into a single block, which is then repeated `K` times.
We also have a one extra block at the beginning and a few more at the end that are invariant of `K` and `R`, and we use CTC loss.

A Jasper model looks like roughly this:

![Jasper with CTC](https://raw.githubusercontent.com/NVIDIA/NeMo/master/docs/sources/source/asr/jasper_vertical.png)

#### Specifying Our Model with a YAML Config File

For this tutorial, we'll build a *Jasper_4x1 model*, with `K=4` blocks of single (`R=1`) sub-blocks and a *greedy CTC decoder*, using the configuration found in `./configs/jasper_an4.yaml`.

If we open up this config file, we find that there is an entry labeled `JasperEncoder`, with a field called `jasper` that contains a list with multiple entries. Each of the members in this list specifies one block in our model, and looks something like this:
```
- filters: 256
  repeat: 1
  kernel: [11]
  stride: [2]
  dilation: [1]
  dropout: 0.2
  residual: false
```
The first member of the list corresponds to the first block in the Jasper architecture diagram, which appears regardless of `K` and `R`.
Next, we have four entries that correspond to the `K=4` blocks, and each has `repeat: 1` since we are using `R=1`.
These are followed by two more entries for the blocks that appear at the end of our Jasper model before the CTC loss.

There are also some entries at the top of the file that specify that we should be shuffling our training data but not our evaluation data (see `AudioToTextDataLayer_train` and `AudioToTextDataLayer_eval`), and some specifications for preprocessing and converting the audio data (in `AudioToMelSpectrogramPreprocessor`).

Using a YAML config such as this is helpful for getting a quick and human-readable overview of what your architecture looks like, and allows you to swap out model and run configurations easily without needing to change your code.

#### Building Training and Evaluation DAGs with NeMo

Building a model using NeMo consists of (1) instantiating the neural modules we need and (2) specifying the DAG by linking them together.

In NeMo, **the training and inference pipelines are managed by a `NeuralModuleFactory`**, which takes care of checkpointing, callbacks, and logs, along with other details in training and inference. We set its `log_dir` argument to specify where our model logs and outputs will be written, and can set other training and inference settings in its constructor. For instance, if we were **resuming training from a checkpoint**, we would set the argument `checkpoint_dir=<path_to_checkpoint>`.

Along with logs in NeMo, you can optionally view the tensorboard logs with the `create_tb_writer=True` argument to the `NeuralModuleFactory`. By default all the tensorboard log files will be stored in `{log_dir}/tensorboard`, but you can change this with the `tensorboard_dir` argument. One can load tensorboard logs through tensorboard by running `tensorboard --logdir=<path_to_tensorboard dir>` in the terminal.
"""

# Create our NeuralModuleFactory, which will oversee the neural modules.
#from nemo.core import DeviceType
#neural_factory = nemo.core.NeuralModuleFactory(
#    log_dir=data_dir+'/an4_tutorial/',
#    create_tb_writer=True,
#    placement=(DeviceType.GPU if torch.cuda.is_available() else DeviceType.CPU))

#logger = nemo.logging
import torch
from nemo.core import DeviceType


neural_factory = nemo.core.NeuralModuleFactory(
    log_dir=data_dir+'/an4_tutorial/',
    placement=(DeviceType.GPU if torch.cuda.is_available() else DeviceType.CPU)
    )

"""Note that if you would like your pipeline to support running on CPU, you should
instead add these imports and then add an additional 'placement' argument:

```python
```

Now that we have our neural module factory, we can **specify our neural modules and instantiate them**. Here, we load the parameters for each module from the configuration file using `import_from_config`. For the module parameters that we can't read directly from the config file or want to overwrite, we can use the `overwrite_params` argument.
"""

# --- Config Information ---#
from ruamel.yaml import YAML

config_path = './configs/jasper_an4.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
labels = params['labels'] # Vocab

# --- Instantiate Neural Modules --- #

# Create training and test data layers (which load data) and data preprocessor
data_layer_train = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer_train",
    overwrite_params={"manifest_filepath": train_manifest}
) # Training datalayer

data_layer_test = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer_eval",
    overwrite_params={"manifest_filepath": test_manifest}
) # Eval datalayer

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor.import_from_config(
    config_path, "AudioToMelSpectrogramPreprocessor"
)

# Create the Jasper_4x1 encoder as specified, and a CTC decoder
encoder = nemo_asr.JasperEncoder.import_from_config(
    config_path, "JasperEncoder"
)

decoder = nemo_asr.JasperDecoderForCTC.import_from_config(
    config_path, "JasperDecoderForCTC",
    overwrite_params={"num_classes": len(labels)}
)

ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

"""The next step is to assemble our training DAG by specifying the inputs to each neural module."""

# --- Assemble Training DAG --- #
audio_signal, audio_signal_len, transcript, transcript_len = data_layer_train()

processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)

encoded, encoded_len = encoder(
    audio_signal=processed_signal,
    length=processed_signal_len)

log_probs = decoder(encoder_output=encoded)
preds = greedy_decoder(log_probs=log_probs)  # Training predictions
loss = ctc_loss(
    log_probs=log_probs,
    targets=transcript,
    input_length=encoded_len,
    target_length=transcript_len)

"""We would like to be able to evaluate our model on the test set, as well, so let's set up the evaluation DAG.

Our **evaluation DAG will reuse most of the parts of the training DAG with the exception of the data layer**, since we are loading the evaluation data from a different file but evaluating on the same model. Note that if we were using data augmentation in training, we would also leave that out in the evaluation DAG.
"""

# --- Assemble Validation DAG --- #
(audio_signal_test, audio_len_test,
 transcript_test, transcript_len_test) = data_layer_test()

processed_signal_test, processed_len_test = data_preprocessor(
    input_signal=audio_signal_test,
    length=audio_len_test)

encoded_test, encoded_len_test = encoder(
    audio_signal=processed_signal_test,
    length=processed_len_test)

log_probs_test = decoder(encoder_output=encoded_test)
preds_test = greedy_decoder(log_probs=log_probs_test)  # Test predictions
loss_test = ctc_loss(
    log_probs=log_probs_test,
    targets=transcript_test,
    input_length=encoded_len_test,
    target_length=transcript_len_test)

"""### Running the Model

We would like to be able to monitor our model while it's training, so we use **callbacks**. In general, *callbacks are functions that are called at specific intervals over the course of training or inference*, such as at the start or end of every *n* iterations, epochs, etc. The callbacks we'll be using for this are the `SimpleLossLoggerCallback`, which reports the training loss (or another metric of your choosing, such as WER for ASR tasks), and the `EvaluatorCallback`, which regularly evaluates the model on the test set. Both of these callbacks require you to pass in the tensors to be evaluated--these would be the final outputs of the training and eval DAGs above.

Another useful callback is the `CheckpointCallback`, for saving checkpoints at set intervals. We create one here just to demonstrate how it works.
"""

# --- Create Callbacks --- #
# We use these imports to pass to callbacks more complex functions to perform.
from nemo.collections.asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch
from functools import partial

train_callback = nemo.core.SimpleLossLoggerCallback(
    # Notice that we pass in loss, predictions, and the transcript info.
    # Of course we would like to see our training loss, but we need the
    # other arguments to calculate the WER.
    tensors=[loss, preds, transcript, transcript_len],
    # The print_func defines what gets printed.
    print_func=partial(
        monitor_asr_train_progress,
        labels=labels),
    tb_writer=neural_factory.tb_writer
    )

# We can create as many evaluation DAGs and callbacks as we want,
# which is useful in the case of having more than one evaluation dataset.
# In this case, we only have one.
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_test, preds_test, transcript_test, transcript_len_test],
    user_iter_callback=partial(
        process_evaluation_batch, labels=labels),
    user_epochs_done_callback=process_evaluation_epoch,
    eval_step=500,  # How often we evaluate the model on the test set
    tb_writer=neural_factory.tb_writer
    )

checkpoint_saver_callback = nemo.core.CheckpointCallback(
    folder=data_dir+'/an4_checkpoints',
    step_freq=1000  # How often checkpoints are saved
    )

if not os.path.exists(data_dir+'/an4_checkpoints'):
    os.makedirs(data_dir+'/an4_checkpoints')

"""Now that we have our model and callbacks set up, **how do we run it**?

Once we create our neural factory and the callbacks for the information that we want to see, we can **start training** by simply calling the train function on the tensors we want to optimize and our callbacks!
"""

# --- Start Training! --- #

neural_factory.train(
    tensors_to_optimize=[loss],
    callbacks=[train_callback, eval_callback, checkpoint_saver_callback],
    optimizer='novograd',
    optimization_params={
        "num_epochs": 100, "lr": 0.01, "weight_decay": 1e-4
    })

# Training for 100 epochs will take a little while, depending on your machine.
# It should take about 20 minutes on Google Colab.
# At the end of 100 epochs, your evaluation WER should be around 20-25%.

"""There we go! We've put together a full training pipeline for the model and trained it for 100 epochs.

### Inference

What if we have a trained model that we **just want to run inference** on?

In that case, we just need to instantiate and link up the modules needed for the evaluation DAG (same procedure as before), and then run `infer` to get the results. Let's see what performing inference with our last checkpoint would look like.
"""

# --- Inference Only --- #

# We've already built the inference DAG above, so all we need is to call infer().
evaluated_tensors = neural_factory.infer(
    # These are the tensors we want to get from the model.
    tensors=[loss_test, preds_test, transcript_test, transcript_len_test],
    # checkpoint_dir specifies where the model params are loaded from.
    checkpoint_dir=(data_dir+'/an4_checkpoints')
    )

# Process the results to get WER
from nemo.collections.asr.helpers import word_error_rate, \
    post_process_predictions, post_process_transcripts

greedy_hypotheses = post_process_predictions(
    evaluated_tensors[1], labels)

references = post_process_transcripts(
    evaluated_tensors[2], evaluated_tensors[3], labels)

wer = word_error_rate(hypotheses=greedy_hypotheses, references=references)
print("*** Greedy WER: {:.2f} ***".format(wer * 100))

"""And that's it!

## Model Improvements

You already have all you need to create your own ASR model in NeMo, but there are a few more tricks that you can employ if you so desire. In this section, we'll briefly cover a few possibilities for improving an ASR model.

### Data Augmentation

There exist several ASR data augmentation methods that can increase the size of our training set.

For example, we can perform augmentation on the spectrograms by zeroing out specific frequency segments ("frequency masking") or time segments ("time masking") as described by [SpecAugment](https://arxiv.org/abs/1904.08779), or zero out rectangles on the spectrogram as in [Cutout](https://arxiv.org/pdf/1708.04552.pdf). In NeMo, we can do all three of these by simply adding in a `SpectrogramAugmentation` neural module. (As of now, it does not perform the time warping from the SpecAugment paper.)
"""

# Create a SpectrogramAugmentation module
spectrogram_aug = nemo_asr.SpectrogramAugmentation(
    rect_masks=5, rect_time=120, rect_freq=50)

# Rearrange training DAG to use augmentation.
# The following code is mostly copy/pasted from the "Assemble Training DAG"
# section, with only one line added!
audio_signal, audio_signal_len, transcript, transcript_len = data_layer_train()

processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)

############## This is the only part that's changed! ##############
processed_signal_aug = spectrogram_aug(input_spec=processed_signal)

encoded, encoded_len = encoder(
    audio_signal=processed_signal_aug,  # Change this argument too
    length=processed_signal_len)
###################################################################

log_probs = decoder(encoder_output=encoded)
preds = greedy_decoder(log_probs=log_probs)  # Training predictions
loss = ctc_loss(
    log_probs=log_probs,
    targets=transcript,
    input_length=encoded_len,
    target_length=transcript_len)

# And then you can train as usual.
# If you want to try it out in this notebook,
# be sure to run neural_factory.reset_trainer() before training again!

"""Another popular method of ASR data augmentation is speed perturbation, where the audio is sped up or slowed down slightly (e.g. 10% faster or slower). See the `SpeedPerturbation` class in the ASR collection for more details.

### Using a Language Model

Though a language model (LM) may not be especially suited to a task like AN4 where we have a bunch of letters being read in sequence, adding a language model for inference can the improve WER in most other ASR tasks, where the speech more closely matches normal patterns. We can use the probability distribution that a language model gives us to better match our predictions to sequences of words we would be more likely to see in the real world, such as correcting "keyboard and house" to "keyboard and mouse."

If you have a language model that you'd like to use with a NeMo model, you can add a `BeamSearchDecoderWithLM` module to your DAG to get beam search predictions that use your language model file.

For the sake of example, even though an LM won't help much for this dataset, we'll go through how to set this up.
First, if you're on your own machine, you'll want to run the script `NeMo/scripts/install_decoders.sh` (or `NeMo/scripts/install_decoders_MacOS.sh`, if appropriate).

**Only run the following code block if you are using Google Colab.**
"""

