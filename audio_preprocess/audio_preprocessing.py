import glob
import os

from .audio_handler import AudioHandler
from scipy.io import wavfile
import pickle
from constants import Constants
from logger import logger
from numba import cuda



def cross_check_existence(audio_fname_list, mesh_fname_list):
    _audio_list =[i.split('/')[-1].replace('.wav', '') for i in audio_fname_list]
    _mesh_list = [i.split('/')[-1] for i in mesh_fname_list]

    miss_sent = set(_audio_list).symmetric_difference(set(_mesh_list))

    for sent in miss_sent:
        audio_fname_list = [i for i in audio_fname_list if sent not in i]
        mesh_fname_list = [i for i in mesh_fname_list if sent not in i]

    assert len(audio_fname_list)+len(miss_sent)==40 and len(mesh_fname_list)+len(miss_sent)==40
    return audio_fname_list, mesh_fname_list 

def process_audio(ds_path, audio, fps):
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1
    if fps == 25:
        config['audio_window_stride'] = 2

    audio_handler = AudioHandler(config)
    processed_audio = audio_handler.process(audio, fps)
    return processed_audio

                                                                        #frame rate, same, do not need change                      # The root of my audios, inside is cliton, obama....
     # names
# audio_list = glob.glob(os.path.join(dataset_path, '*/audio/*.wav'))


# print 'Loading audio for preprocessing...'

def make_pickle(uid, selected_model_str):
    logger(f"Audio preprocessing is started. | Request from {uid} | Processing {selected_model_str}", level="EXE_START")
    fps = 30
    #subjects = ['audio']
    audio4deepspeech = {}
    #for subject in subjects:
    #audio_list = glob.glob(os.path.join(f"{Constants.DEFAULT_USER_DIRECTORY}/{uid}/{uid}_converted.wav"))      # subject file location
    tmp_audio = {}
    #for audio_fname in audio_list:
    #sentence = audio_fname.split('/')[-1][0:-4]                             # get wav name
    sentence = f"{uid}"
    sample_rate, audio = wavfile.read(f"{Constants.DEFAULT_USER_DIRECTORY}/{uid}/{uid}.wav")                          # read wav file
    tmp_audio[sentence] = {'audio':audio, 'sample_rate': sample_rate}

    audio4deepspeech['audio'] = tmp_audio                                       # save format names(obama), video(0-ZCDAUSH)
    logger(f"WAV file successfully loaded. | Request from {uid} | Processing {selected_model_str}", level="INFO")
    ds_fname = Constants.DEFAULT_NEED_FILES + "/" + selected_model_str + "/" + Constants.DEFAULT_AUDIO2FACE_FILE         # deep speech model
    processed_audio = process_audio(ds_fname, audio4deepspeech, fps)                # generate audio feature                             

    out_path = f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}/audio_preprocessed/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    subject = list(processed_audio.keys())[0]
    sentence = list(processed_audio['audio'].keys())[0]
    out_file = os.path.join(out_path, sentence +'.pkl')
    _audio = processed_audio[subject][sentence]['audio'] 

    pickle.dump(_audio, open(out_file, 'wb'))
            
    logger(f"Audio preprocessing is successfully finished. | Request from {uid} | Processing {selected_model_str}", level="EXE_FINISH")
    
    device = cuda.get_current_device()
    device.reset()
    del processed_audio
    logger(f"GPU Memory Reset (Audio Preprocessing) | Request from {uid} | Processing {selected_model_str}", level="INFO")

    return out_file