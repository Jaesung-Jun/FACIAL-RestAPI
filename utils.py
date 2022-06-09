import base64
import os
from scipy.io import wavfile
import ffmpeg
import librosa
import shutil
from constants import Constants
import re
import logger

class Utils:
    
    @staticmethod
    def b64ToBinary(b64):

        b64_new = b64 + '=' * (4 - len(b64) % 4)
        binary = base64.b64decode(b64_new)
        return binary

    @staticmethod
    def binaryToB64(binary):
        with open(binary, "rb") as f:
            data = f.read()
            f.close()
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def removeBase64Header(b64):
        b64_header_removed = re.sub('^data:.*base64,', "", b64)
        return b64_header_removed
        

    @staticmethod
    def loadWavfile(path):
        samplerate, data = wavfile.read(path)
        return samplerate, data
    
    @staticmethod
    def saveFileToDirectory(binary, path):
        with open(path, "wb") as f:
            f.write(binary)

    @staticmethod
    def createFolder(name):
        try:
            if not os.path.exists(name):
                os.makedirs(name)
        except OSError:
            print(f'{name} direcory is existed')

    @staticmethod
    def isExistsUserDirectory(uid):
        return os.path.exists(f"{Constants.DEFAULT_USER_DIRECTORY}/{uid}")

    @staticmethod
    def rmUserDirectory(uid):
        shutil.rmtree(f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}')
    
class Video_Utils:
    @staticmethod
    def concatAudioVideo(audio_path, video_path):

        video = ffmpeg.input(video_path)
        audio = ffmpeg.input(audio_path)
        save_path = os.path.splitext(video_path)[0] + ".mp4"
        ffmpeg.concat(video, audio, v=1, a=1).output(save_path, loglevel='error').run()
        
class Audio_Utils:
    @staticmethod
    def convertWavSample(audio_path, output_path, overwrite=True):
        
        input_audio = ffmpeg.input(audio_path)
        ffmpeg.output(input_audio, output_path, loglevel='error', **{'acodec':'pcm_s16le', 'f':'wav', 'ac':'1', 'ar':'16000'}).run()
        if overwrite:
            os.remove(audio_path)
            os.rename(output_path, audio_path)

    @staticmethod
    def getWavLength(audio_path):
        duration = librosa.get_duration(filename=audio_path)
        return duration