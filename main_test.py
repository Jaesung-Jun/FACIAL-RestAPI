from error import ErrorMessages
from utils import Utils, Video_Utils, Audio_Utils
from test_audio import test_audio
from audio_preprocess import audio_preprocessing
from face_rendering import rendering_gaosi
from constants import Constants
from face2vid import test_video
import os
from logger import logger

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main_test(uid, voice, selected_model):

    if Utils.isExistsUserDirectory(uid):
        Utils.rmUserDirectory(uid)
        logger(f"Remove user directory. | Request from {uid}", level="INFO")

    Utils.createFolder(Constants.DEFAULT_USER_DIRECTORY + "/" + uid)
    request_voice = Utils.b64ToBinary(voice)
    Utils.saveFileToDirectory(request_voice, Constants.DEFAULT_USER_DIRECTORY + "/" + uid + f"/{uid}.wav")
    #Audio_Utils.convertWavSample(uid, Constants.DEFAULT_USER_DIRECTORY + "/" + uid + f"/{uid}.wav", Constants.DEFAULT_USER_DIRECTORY + "/" + uid + f"/{uid}_new.wav")
    #print(Audio_Utils.getWavLength(Constants.DEFAULT_USER_DIRECTORY + "/" + uid + f"/{uid}.wav"))

    if selected_model == Constants.ADJ:
        selected_model_str = "ADJ"
        pkl_path = audio_preprocessing.make_pickle(uid, selected_model_str)
        npz_path = test_audio.test_audio(pkl_path, uid, selected_model_str)
        rendering_gaosi.rendering(uid, selected_model_str, npz_path)
        test_video.test_video(uid, selected_model_str)
        
    elif selected_model == Constants.JJS:
        selected_model_str = "JJS"
        audio_preprocessing.make_pickle('FACIAL_JJS')
        test_audio.test_audio('FACIAL_JJS')
        rendering_gaosi.rendering('FACIAL_JJS')
        
    elif selected_model == Constants.KSK:
        selected_model_str = "KSK"
        audio_preprocessing.make_pickle('FACIAL_KSK')
        test_audio.test_audio('FACIAL_KSK')
        rendering_gaosi.rendering('FACIAL_KSK')
        
    elif selected_model == Constants.MJH:
        selected_model_str = "MJH"
        audio_preprocessing.make_pickle('FACIAL_MJH')
        test_audio.test_audio('FACIAL_MJH')
        rendering_gaosi.rendering('FACIAL_MJH')
        
    elif selected_model == Constants.SJG:
        selected_model_str = "SJG"
        audio_preprocessing.make_pickle('FACIAL_SJG')
        test_audio.test_audio('FACIAL_SJG')
        rendering_gaosi.rendering('FACIAL_SJG')
    else:
        print("Error!")

    Video_Utils.concatAudioVideo(f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}/{uid}.wav', f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}/complete_{uid}.avi')

#_, wave_file = Utils.load_wavfile("./backend-server/Another_One_Bites_The_Dust.wav")
b64_str = Utils.binaryToB64("./test_wave_files/Another_One_Bites_The_Dust.wav")
main_test("2000", b64_str, Constants.ADJ)
#Audio_Utils.convertWavSample("./test_wave2.wav")