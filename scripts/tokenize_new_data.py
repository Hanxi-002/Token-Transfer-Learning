# we are going to process all the midi files in this folder:
# new_data_input_folder = '/ix/djishnu/Aaron/2_misc/PGM_Project/content/clean_midi'
new_data_input_folder = '/ix/djishnu/Aaron/2_misc/PGM_Project/content/lakh_data'

# new_data_output_folder = '/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/clean_midi'
new_data_output_folder = '/ix/djishnu/Aaron/2_misc/PGM_Project/content_tokenizing/lakh_tokenized'

#@title Import all needed modules

print('Loading needed modules. Please wait...')

import sys
import os
import glob
sys.path.insert(1, '/ix/djishnu/Aaron/2_misc/PGM_Project/Full-MIDI-Music-Transformer')


import math
import statistics
import random
import pickle
from tqdm import tqdm
import pandas as pd

print('Loading TMIDIX module...')

import TMIDIX


print('Done!')
print('Enjoy! :)')

output_dir = new_data_output_folder

input_dir = new_data_input_folder

pickle_file_name = output_dir + "/" + "20231206_midis"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Loading MIDI files...')
print('This may take a while on a large dataset in particular.')

filez = list()
for (dirpath, dirnames, filenames) in os.walk(input_dir):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

if filez == []:
    print('Could not find any MIDI files. Please check Dataset dir...')
    print('=' * 70)

print('Randomizing file list...')
random.shuffle(filez)


# write file to pickle
TMIDIX.Tegridy_Any_Pickle_File_Writer(filez, pickle_file_name)


process_output_dir = output_dir + "/"

#@title Process MIDIs with TMIDIX MIDI processor
print('=' * 70)
print('TMIDIX MIDI Processor')
print('=' * 70)
print('Starting up...')
print('=' * 70)

###########

START_FILE_NUMBER = 0
LAST_SAVED_BATCH_COUNT = 0

input_files_count = START_FILE_NUMBER
files_count = LAST_SAVED_BATCH_COUNT

melody_chords_f = []

stats = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print('Processing MIDI files. Please wait...')
print('=' * 70)

for f in tqdm(filez[START_FILE_NUMBER:10000]):
    try:

      input_files_count += 1

      fn = os.path.basename(f)

      # Filtering out giant MIDIs
      file_size = os.path.getsize(f)

      if file_size < 250000:

        #=======================================================
        # START PROCESSING

        # Convering MIDI to ms score with MIDI.py module
        ms_score = TMIDIX.midi2single_track_ms_score(open(f, 'rb').read(), recalculate_channels=False)

        events_matrix1 = []

        itrack = 1

        events_types = ['note',
                        'patch_change',
                        'control_change',
                        'key_after_touch',
                        'channel_after_touch',
                        'pitch_wheel_change']

        while itrack < len(ms_score):
            for event in ms_score[itrack]:
                if event[0] in events_types:
                    events_matrix1.append(event)
            itrack += 1

        events_matrix1.sort(key = lambda x: (x[4] if x[0] == 'note' else x[1]), reverse = True)
        events_matrix1.sort(key = lambda x: x[1])

        if len(events_matrix1) > 0:
            if min([e[1] for e in events_matrix1]) >= 0 and min([e[2] for e in events_matrix1 if e[0] == 'note']) >= 0:

                #=======================================================
                # PRE-PROCESSING

                # recalculating timings
                for e in events_matrix1:
                    e[1] = int(e[1] / 16) # Max 2 seconds for start-times
                    if e[0] == 'note':
                        e[2] = int(e[2] / 32) # Max 4 seconds for durations

                #=======================================================
                # FINAL PRE-PROCESSING

                patch_list = [0] * 16
                patch_list[9] = 128

                melody_chords = []

                pe = events_matrix1[0]

                for e in events_matrix1:

                    if e[0] == 'note':

                        # Cliping all values...
                        time = max(0, min(127, e[1]-pe[1]))
                        dur = max(1, min(127, e[2]))
                        cha = max(0, min(15, e[3]))
                        ptc = max(1, min(127, e[4]))
                        vel = max(1, min(127, e[5]))
                        pat = patch_list[cha]

                        # Writing final note
                        melody_chords.append(['note', time, dur, cha, ptc, vel, pat])

                    if e[0] == 'patch_change':

                        # Cliping all values...
                        time = max(0, min(127, e[1]-pe[1]))
                        cha = max(0, min(15, e[2]))
                        ptc = max(0, min(127, e[3]))

                        if cha != 9:
                            patch_list[cha] = ptc
                        else:
                            patch_list[cha] = ptc+128

                        melody_chords.append(['patch_change', time, cha, ptc])

                    if e[0] == 'control_change':

                        # Cliping all values...
                        time = max(0, min(127, e[1]-pe[1]))
                        cha = max(0, min(15, e[2]))
                        con = max(0, min(127, e[3]))
                        cval = max(0, min(127, e[4]))

                        pat = patch_list[cha]

                        melody_chords.append(['control_change', time, pat, con, cval])

                    if e[0] == 'key_after_touch':

                        # Cliping all values...
                        time = max(0, min(127, e[1]-pe[1]))
                        cha = max(0, min(15, e[2]))
                        ptc = max(1, min(127, e[3]))
                        vel = max(1, min(127, e[4]))

                        pat = patch_list[cha]

                        melody_chords.append(['key_after_touch', time, pat, ptc, vel])

                    if e[0] == 'channel_after_touch':

                        # Cliping all values...
                        time = max(0, min(127, e[1]-pe[1]))
                        cha = max(0, min(15, e[2]))
                        vel = max(1, min(127, e[3]))

                        pat = patch_list[cha]

                        melody_chords.append(['channel_after_touch', time, pat, vel])

                    if e[0] == 'pitch_wheel_change':

                        # Cliping all values...
                        time = max(0, min(127, e[1]-pe[1]))
                        cha = max(0, min(15, e[2]))
                        wheel = max(-8192, min(8192, e[3])) // 128

                        pat = patch_list[cha]

                        melody_chords.append(['pitch_wheel_change', time, pat, wheel])

                    pe = e


                #=======================================================

                # Adding SOS/EOS, intro and counters

                if len(melody_chords) < (127 * 100) and ((events_matrix1[-1][1] * 16) < (8 * 60 * 1000)): # max 12700 MIDI events and max 8 min per composition

                    melody_chords1 = [['start', 0, 0, 0, 0, 0]]

                    events_block_counter = 0
                    time_counter = 0

                    for i in range(len(melody_chords)):
                        melody_chords1.append(melody_chords[i])

                        time_counter += melody_chords[i][1]

                        if i != 0 and (len(melody_chords) - i == 100):
                            melody_chords1.append(['outro', 0, 0, 0, 0, 0])

                        if i != 0 and (i % 100 == 0) and (len(melody_chords) - i >= 100):
                            melody_chords1.append(['counters_seq', ((time_counter * 16) // 3968), events_block_counter, 0, 0, 0])
                            events_block_counter += 1

                    melody_chords1.append(['end', 0, 0, 0, 0, 0])

                    #=======================================================

                    melody_chords2 = []

                    for m in melody_chords1:

                        if m[0] == 'note':

                            if m[3] == 9:
                                ptc = m[4] + 128
                            else:
                                ptc = m[4]

                            # Writing final note
                            melody_chords2.extend([m[6], m[1]+256, m[2]+256+128, ptc+256+128+128, m[5]+256+128+128+256])

                        # Total tokens so far 896

                        if m[0] == 'patch_change': # 896

                            melody_chords2.extend([1554, m[1]+256, m[2]+256+128+128+256+128, m[3], 1553])

                        # Total tokens so far 912

                        if m[0] == 'control_change': # 912

                            melody_chords2.extend([1555, m[1]+256, m[2], m[3]+256+128+128+256+128+16, m[4]+256+128+128+256+128+16+128])

                        # Total tokens so far 1168

                        if m[0] == 'key_after_touch': # 1168

                            if m[2] == 9:
                                ptc = m[3] + 128
                            else:
                                ptc = m[3]

                            melody_chords2.extend([1556, m[1]+256, m[2], ptc+256+128+128, m[4]+256+128+128+256])

                        # Total tokens so far 1168

                        if m[0] == 'channel_after_touch': # 1168

                            melody_chords2.extend([1557, m[1]+256, m[2], m[3]+256+128+128+256, 1553])

                        # Total tokens so far 1168

                        if m[0] == 'pitch_wheel_change': # 1168

                            melody_chords2.extend([1558, m[1]+256, m[2], m[3]+256+128+128+256+128+16+128, 1553])

                        # Total tokens so far 1296

                        if m[0] == 'counters_seq': # 1296

                            melody_chords2.extend([1559, m[1]+256+128+128+256+128+16+128+128, m[2]+256+128+128+256+128+16+128+128+128, 1553, 1553])

                        # Total tokens so far: 1552

                        #=======================================================

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

                        if m[0] == 'outro':
                            melody_chords2.extend([1560, 1560, 1560, 1560, 1560])

                        if m[0] == 'end':
                            melody_chords2.extend([1561, 1561, 1561, 1561, 1561])

                        if m[0] == 'start':
                            melody_chords2.extend([1562, 1562, 1562, 1562, 1562])

                    #=======================================================

                    # FINAL TOTAL TOKENS: 1562

                    #=======================================================

                    melody_chords_f.append(melody_chords2)

                    #=======================================================

                    # Processed files counter
                    files_count += 1

                    # Saving every 5000 processed files
                    if files_count % 5000 == 0:
                      print('SAVING !!!')
                      print('=' * 70)
                      print('Saving processed files...')
                      print('=' * 70)
                      print('Processed so far:', files_count, 'out of', input_files_count, '===', files_count / input_files_count, 'good files ratio')
                      print('=' * 70)
                      count = str(files_count)
                      TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, process_output_dir+ '/tokenized_midi' +count)
                      melody_chords_f = []
                      print('=' * 70)

    except KeyboardInterrupt:
        print('Saving current progress and quitting...')
        break

    except Exception as ex:
        print('WARNING !!!')
        print('=' * 70)
        print('Bad MIDI:', f)
        print('Error detected:', ex)
        print('=' * 70)
        continue

# Saving last processed files...
print('SAVING !!!')
print('=' * 70)
print('Saving processed files...')
print('=' * 70)
print('Processed so far:', files_count, 'out of', input_files_count, '===', files_count / input_files_count, 'good files ratio')
print('=' * 70)
count = str(files_count)
TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, process_output_dir+ '/tokenized_midi'+count)

# Displaying resulting processing stats...
print('=' * 70)
print('Done!')
print('=' * 70)

print('Resulting Stats:')
print('=' * 70)
print('Total good processed MIDI files:', files_count)
print('=' * 70)