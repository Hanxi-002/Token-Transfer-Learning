#!/bin/python

import os
os.chdir('/ix/djishnu/Aaron/2_misc/PGM_Project')

out_dir = 'outputs/improvised_4000token_cold_start/midi'
#@markdown If first note MIDI patch number == -1 the model will generate first note itself

first_note_MIDI_patch_number = -1 # @param {type:"slider", min:-1, max:128, step:1}

number_of_runs = 2

number_of_tokens_tp_generate = 4000 # @param {type:"slider", min:30, max:4095, step:5}



import sys

sys.path.insert(1, '/ix/djishnu/Aaron/2_misc/PGM_Project/Full-MIDI-Music-Transformer')

import TMIDIX
from x_transformer import *


#@title Import modules

print('=' * 70)
print('Loading core Full MIDI Music Transformer modules...')

import pickle
import secrets
import statistics
from time import time
import tqdm

from huggingface_hub import hf_hub_download

print('=' * 70)
print('Loading main Full MIDI Music Transformer modules...')
import torch

# %cd /content/Full-MIDI-Music-Transformer

print('=' * 70)
print('Loading aux Full MIDI Music Transformer modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from midi2audio import FluidSynth
from IPython.display import Audio, display

import random

# from google.colab import files

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

#@title Load Full MIDI Music Transformer Tiny Model (FAST)

#@markdown Very fast model, 16 layers, 225k MIDIs training corpus

full_path_to_model_checkpoint = "Full-MIDI-Music-Transformer/Models/Tiny/Full_MIDI_Music_Transformer_Tiny_Trained_Model_30000_steps_0.2859_loss_0.9167_acc.pth" #@param {type:"string"}

#@markdown Model precision option

model_precision = "bfloat16-float16" # @param ["bfloat16-float16", "float32"]

#@markdown bfloat16-float16 == Half precision/double speed

#@markdown float32 == Full precision/normal speed

plot_tokens_embeddings = True # @param {type:"boolean"}

print('=' * 70)
print('Loading Full MIDI Music Transformer Tiny Pre-Trained Model...')
print('Please wait...')
print('=' * 70)

if os.path.isfile(full_path_to_model_checkpoint):
  print('Model already exists...')

else:
  hf_hub_download(repo_id='asigalov61/Full-MIDI-Music-Transformer',
                  filename='Full_MIDI_Music_Transformer_Tiny_Trained_Model_30000_steps_0.2859_loss_0.9167_acc.pth',
                  local_dir='Full-MIDI-Music-Transformer/Models/Tiny/',
                  local_dir_use_symlinks=False)
print('=' * 70)
print('Instantiating model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'
# device_type = 'cpu'

if model_precision == 'bfloat16-float16':
  dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
else:
  dtype = 'float32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

SEQ_LEN = 4096

# instantiate the model

model = TransformerWrapper(
    num_tokens = 1564,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 18, heads = 8)
)

model = AutoregressiveWrapper(model, ignore_index = 1563)

model = torch.nn.DataParallel(model)

model.cuda()
print('=' * 70)

print('Loading model checkpoint...')

# load gpu model
model.load_state_dict(torch.load(full_path_to_model_checkpoint))

# load cpu model
# model.load_state_dict(torch.load(full_path_to_model_checkpoint, map_location=torch.device('cpu')))


print('=' * 70)

model.eval()

print('Done!')
print('=' * 70)
print('Model will use', dtype, 'precision')
print('=' * 70)

# Model stats
print('Model summary...')
summary(model)


#@title Standard Improv Generator

#@markdown Improv settings

# #@markdown If first note MIDI patch number == -1 the model will generate first note itself

# first_note_MIDI_patch_number = -1 # @param {type:"slider", min:-1, max:128, step:1}

# number_of_runs = 100
#@markdown Generation settings

# number_of_tokens_tp_generate = 4000 # @param {type:"slider", min:30, max:4095, step:5}
number_of_batches_to_generate = 16 #@param {type:"slider", min:1, max:16, step:1}
temperature = 0.9 #@param {type:"slider", min:0.1, max:1, step:0.1}

#@markdown Other settings

render_MIDI_to_audio = False # @param {type:"boolean"}

print('=' * 70)
print('Full MIDI Music Transformer Improv Model Generator')
print('=' * 70)

outy = [1562, 1562, 1562, 1562, 1562]

if first_note_MIDI_patch_number > -1:
  outy.extend([first_note_MIDI_patch_number, 0+256, 16+256+128, 72+256+128+128, 90+256+128+128+256])

print('Selected Improv Sequence:')
print(outy[:10])
print('=' * 70)

inp = [outy] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()

with ctx:
  out = model.module.generate(inp,
                        number_of_tokens_tp_generate,
                        temperature=temperature,
                        return_prime=True,
                        verbose=True)

out0 = out.tolist()

print('=' * 70)
print('Done!')
print('=' * 70)

#======================================================================

print('Rendering results...')
for run_num in range(number_of_runs):
  for i in range(number_of_batches_to_generate):

    print('=' * 70)
    print('Batch #', i)
    print('=' * 70)

    out1 = out0[i]

    print('Sample INTs', out1[:10])
    print('=' * 70)

    if len(out1) != 0:

        song = out1
        song_f = []

        time = 0
        dur = 0
        vel = 90
        pitch = 0
        channel = 0

        son = []
        song1 = []
        for j in range(0, len(song), 5): # creating penta seqs...
            song1.append(song[j:j+5])

        patch_list = [0] * 16
        patch_list[9] = 128

        channels_list = [0] * 16
        channels_list[9] = 1

        for s in song1: # decoding...

            # 1553 - pad token

            # 1554 - patch change token
            # 1555 - control change token
            # 1556 - key after touch token
            # 1557 - channel after touch token
            # 1558 - pitch wheel change token
            # 1559 - counters seq token

            # 1560 - outro token
            # 1561 - end token
            # 1562 - start token

            if s[0] < 256: # Note

                patch = s[0]
                time += (s[1]-256) * 16
                dur = (s[2]-256-128) * 32
                pitch = (s[3]-256-128-128) % 128
                vel = (s[4]-256-128-128-256)

                if patch in patch_list:
                    channel = patch_list.index(patch)
                    channels_list[channel] = 1

                else:
                    if 0 in channels_list:
                        channel = channels_list.index(0)
                        channels_list[channel] = 1
                        song_f.append(['patch_change', time, channel, patch])

                    else:
                        channel = 15
                        channels_list[channel] = 1
                        song_f.append(['patch_change', time, channel, patch])

                song_f.append(['note', time, dur, channel, pitch, vel])

            if s[0] == 1554: # patch change

                time += (s[1]-256) * 16
                channel = (s[2]-(256+128+128+256+128))
                patch = s[3]

                if channel != 9:
                    patch_list[channel] = patch
                else:
                    patch_list[channel] = patch + 128

                song_f.append(['patch_change', time, channel, patch])

            if s[0] == 1555: # control change

                time += (s[1]-256) * 16
                patch = s[2]
                controller = (s[3]-(256+128+128+256+128+16))
                controller_value = (s[4]-(256+128+128+256+128+16+128))

                try:
                    channel = patch_list.index(patch)
                except:
                    channel = 15

                song_f.append(['control_change', time, channel, controller, controller_value])

            if s[0] == 1556: # key after touch

                time += (s[1]-256) * 16
                patch = s[2]
                pitch = (s[3]-256-128-128) % 128
                vel = (s[4]-256-128-128-256)

                try:
                    channel = patch_list.index(patch)
                except:
                    channel = 15

                song_f.append(['key_after_touch', time, channel, pitch, vel])

            if s[0] == 1557: # channel after touch

                time += (s[1]-256) * 16
                patch = s[2]
                vel = (s[3]-256-128-128-256)

                try:
                    channel = patch_list.index(patch)
                except:
                    channel = 15

                song_f.append(['channel_after_touch', time, channel, vel])

            if s[0] == 1558: # pitch wheel change

                time += (s[1]-256) * 16
                patch = s[2]
                pitch_wheel = (s[3]-(256+128+128+256+128+16+128)) * 128

                try:
                    channel = patch_list.index(patch)
                except:
                    channel = 15

                song_f.append(['pitch_wheel_change', time, channel, pitch_wheel])

        detailed_stats = TMIDIX.Tegridy_SONG_to_Full_MIDI_Converter(song_f,
                                                                    output_signature = 'Full MIDI Music Transformer',
                                                                    output_file_name = out_dir + '/composition_run_'+ str(run_num) + "_" +str(i),
                                                                    track_name='Project Los Angeles'
                                                                    )

        print('=' * 70)
        print('Displaying resulting composition...')
        print('=' * 70)

        fname = '/Full-MIDI-Music-Transformer-Composition_'+str(i)

        x = []
        y =[]
        c = []

        colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'pink', 'orange', 'purple', 'gray', 'white', 'gold', 'silver', 'red', 'yellow', 'green', 'cyan']

        for s in song_f:
            if s[0] == 'note':
                x.append(s[1] / 1000)
                y.append(s[4])
                c.append(colors[s[3]])

        if render_MIDI_to_audio:
            FluidSynth().midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))

            # FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))

            display(Audio(str(fname + '.wav'), rate=16000))

        plt.figure(figsize=(14,5))
        ax=plt.axes(title=fname)
        ax.set_facecolor('black')

        plt.scatter(x,y, c=c)
        plt.xlabel("Time")
        plt.ylabel("Pitch")
        plt.show()

