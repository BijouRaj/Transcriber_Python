import librosa # audio analysis, onset detection, pitch tracking, note conversion
#import librosa.display 
import numpy as np 

# load file
audio_file = 'cmajor.ogg'
# y is audio time series, sr is sampling rate
y, sr = librosa.load(audio_file)

# Onset detection, frame by frame
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
# converts frame numbers to time in seconds, frame length is 512 samples
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

# Insert manual onset at beginning of track in case it was missed
if len(onset_frames) == 0 or onset_frames[0] != 0:
    onset_frames = np.insert(onset_frames, 0, 0)
    onset_times = np.insert(onset_times, 0, 0.0)

# Pitch estimation using PYIN, monophonic pitch tracker
# f0 is fundamental frequency at each frame, voiced_flag is note vs silence, voiced_probs is probability that frame contains pitch
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    # limit pitch from C2 to C7 to avoid false detections
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7')
)

# Convert Hz into MIDI note numbers
midi = librosa.hz_to_midi(f0)

# Extract note at each onset
notes = []
for i, onset in enumerate(onset_frames[:-1]):  # skip last, handled separately
    # determine start frame of note
    start_frame = onset
    # window contains only one note
    end_frame = onset_frames[i+1]

    # Take the most common MIDI value in this window
    midi_segment = midi[start_frame:end_frame]
    # remove frames with no pitch, empty 
    midi_segment = midi_segment[~np.isnan(midi_segment)] 

    # if pitch detected
    if len(midi_segment) > 0:
        # take median note
        midi_note = int(np.round(np.median(midi_segment)))  
        # convert MIDI number to note name
        note_name = librosa.midi_to_note(midi_note) 
        # determine duration of note
        duration = onset_times[i+1] - onset_times[i]
        # add each to notes to be printed at the end
        notes.append((midi_note, onset, duration, note_name))

# Handle the last note until end of file
last_onset = onset_frames[-1]
last_time = librosa.frames_to_time(last_onset, sr=sr, hop_length=512)
midi_segment = midi[last_onset:]
midi_segment = midi_segment[~np.isnan(midi_segment)]

if len(midi_segment) > 0:
    midi_note = int(np.round(np.median(midi_segment)))
    note_name = librosa.midi_to_note(midi_note)
    duration = librosa.get_duration(y=y, sr=sr) - last_time
    notes.append((midi_note, last_onset, duration, note_name))

# Print results
print("MIDI note \t Onset frame \t Duration (s) \t Note")
for entry in notes:
    print(entry[0], '\t\t', entry[1], '\t\t', f"{entry[2]:.2f}", '\t\t', entry[3])
