
import librosa

audio_file = 'cmajor.ogg'
y, sr = librosa.load(audio_file)

def translate(pitch):
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = notes[pitch]
    return note

chroma = librosa.feature.chroma_stft(y = y, sr = sr)
onset_frames = librosa.onset.onset_detect(y = y, sr = sr)

prev_time = None
notes = [] 
for onset in onset_frames:
    chroma_at_onset = chroma[:, onset] 
    note_pitch = chroma_at_onset.argmax() 
    onset_time = librosa.frames_to_time(onset, sr = sr, hop_length=512)
    
    if prev_time != None:
        duration = onset_time - prev_time 
        notes.append((note_pitch, onset, duration, translate(note_pitch)))
        #note_duration = librosa.frames_to_time(onset, sr=sr)
        #notes.append((note_pitch, onset, note_duration - prev_note_duration, translate(note_pitch)))
        #prev_note_duration = note_duration 
    prev_time = onset_time 
    #else:
    #    prev_note_duration = librosa.frames_to_time(onset, sr=sr)
    #    first = False 
if prev_time != None:
    total_time = librosa.get_duration(y=y, sr=sr)
    duration = total_time - prev_time 
    last_pitch = chroma[:, onset_frames[-1]].argmax()
    notes.append((last_pitch, onset, duration, translate(last_pitch)))

print("Note pitch \t Onset frame \t Note duration \t Note")
for entry in notes:
    print(entry[0], '\t\t', entry[1], '\t\t', entry[2], "\t\t", entry[3])