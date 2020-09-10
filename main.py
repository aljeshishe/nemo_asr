import glob
import os
import subprocess
import sys
import tarfile
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import wget
import random
import numpy as np
import torch
from nemo.core import DeviceType
from callbacks import WandbLogger
import nemo
import nemo.collections.asr as nemo_asr
import json
import librosa
from ruamel.yaml import YAML
from nemo.collections.asr import helpers
from functools import partial

parser = ArgumentParser()
parser.add_argument("name", nargs="?", default="nemo_asr", help="run name")
args = parser.parse_args()

if torch.cuda.is_available():
    eval_freq = 100
else:
    eval_freq = 1

def seed_torch(seed=1029):
    print(f'seeding {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

print("Download the dataset. This will take a few moments...")
print("******")
data_dir = '.'
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


print('Building Manifest Files')
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



neural_factory = nemo.core.NeuralModuleFactory(
    log_dir=data_dir+'/an4_tutorial/',
    placement=(DeviceType.GPU if torch.cuda.is_available() else DeviceType.CPU),
    random_seed=42,
    )

print('Config Information')

config_path = './configs/jasper_an4.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
labels = params['labels'] # Vocab

print('Instantiate Neural Modules')

# Create training and test data layers (which load data) and data preprocessor
data_layer_train = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer_train",
    overwrite_params={"manifest_filepath": train_manifest},
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

print('Assemble Training DAG')
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


print('Assemble Validation DAG')
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


print('Create Callbacks')
callbacks = []
cb = nemo.core.SimpleLogger(step_freq=1)
callbacks.append(cb)
os.environ["WANDB_API_KEY"] = "5c5f03d42e16ce3df7aaabb404480128adef6719"
runid = datetime.now().strftime("%H%M%S")
wandb_name = f'{args.name}_{runid}'
cb = WandbLogger(
    step_freq=1, runid=runid,
    folder=Path("run") / runid, name=wandb_name,
    save_freq=1000,
    asr_model=None,
    args=dict(a=1, b=2)
)
callbacks.append(cb)

cb = nemo.core.SimpleLossLoggerCallback(
    # Notice that we pass in loss, predictions, and the transcript info.
    # Of course we would like to see our training loss, but we need the
    # other arguments to calculate the WER.
    tensors=[loss, preds, transcript, transcript_len],
    # The print_func defines what gets printed.
    print_func=partial(
        helpers.monitor_asr_train_progress,
        labels=labels),
    )
callbacks.append(cb)

# We can create as many evaluation DAGs and callbacks as we want,
# which is useful in the case of having more than one evaluation dataset.
# In this case, we only have one.
cb = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_test, preds_test, transcript_test, transcript_len_test],
    user_iter_callback=partial(helpers.process_evaluation_batch, labels=labels),
    user_epochs_done_callback=helpers.process_evaluation_epoch,
    eval_step=eval_freq,  # How often we evaluate the model on the test set
    tb_writer=neural_factory.tb_writer,
    wandb_name=wandb_name,
    wandb_project='asr',
)
callbacks.append(cb)

# cb = nemo.core.CheckpointCallback(folder=data_dir+'/an4_checkpoints', step_freq=1000)
# callbacks.append(cb)
os.makedirs(data_dir+'/an4_checkpoints', exist_ok=True)

# seed_torch(42)

print('Start Training!')
neural_factory.train(
    tensors_to_optimize=[loss],
    callbacks=callbacks,
    optimizer='novograd',
    optimization_params={
        "num_epochs": 20, "lr": 0.01, "weight_decay": 1e-4
    })

print('Inference Only')
# We've already built the inference DAG above, so all we need is to call infer().
evaluated_tensors = neural_factory.infer(
    # These are the tensors we want to get from the model.
    tensors=[loss_test, preds_test, transcript_test, transcript_len_test],
    # checkpoint_dir specifies where the model params are loaded from.
    checkpoint_dir=(data_dir+'/an4_checkpoints')
    )

# Process the results to get WER
greedy_hypotheses = helpers.post_process_predictions(
    evaluated_tensors[1], labels)

references = helpers.post_process_transcripts(
    evaluated_tensors[2], evaluated_tensors[3], labels)

wer = helpers.word_error_rate(hypotheses=greedy_hypotheses, references=references)
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

