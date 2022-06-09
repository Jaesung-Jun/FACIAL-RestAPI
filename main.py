from flask import Flask, jsonify, request
from flask_cors import CORS

from error import ErrorMessages
from utils import Utils, Video_Utils, Audio_Utils
from constants import Constants

from test_audio import test_audio
from audio_preprocess import audio_preprocessing
from face_rendering import rendering_gaosi
from face2vid import test_video

from logger import logger
 
import os

import faulthandler
faulthandler.enable()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

"""##########################################################################################"""
"""                                       테스트 순서                                        """
"""                                                                                          """
""" 1. voice파일을 audio_preprocessing의 make_pickle을 사용해서 pkl파일로 변환               """
""" 2. test의 test_audio을 사용해서 1번에서 만든 pkl파일을 npz로 변환                        """
""" 3. rendering_gaosi의 rendering을 사용해서 2번에서 만든 npz파일을 사진으로 렌더링해서 저장"""
""" 4. 3번에서 만든 사진을 이용해서 비디오로 인코딩함                                        """
"""##########################################################################################"""

"""1. make_pickle함수에서 audio_list변수의 값(경로)을 프론트에서 보내주는 wav파일 있는 곳으로 바꿔야함.
   2. out_path변수의 값(경로)을 uid폴더로 바꿔서 pkl파일 저장"""

"""1. test_audio함수에서 audiopath변수의 값(경로)을 pkl이 들어있는 uid폴더로 바꿔야함
   2. out_path변수의 값(경로)을 uid폴더로 바꿔서 npz파일 저장"""

"""1. rendering함수에서 net_params_path변수의 값(경로)을 npz파일이 들어있는 uid폴더로 바꿔야함
   2. outpath변수의 값(경로)을 uid폴더로 바꿔서 렌더링한 사진 저장"""

"""1. testvideo함수에서 경로 설정 해야됨"""

# ! 필요한 파일 목록
# ? 1. eyemasy.npy
# ? 2. Gen-10.mdl
# ? 3. /video_preprocess/train1_posenew.npz
# ? 4. /examples/test-result/test1.npz
# TODO 1. ./files/{selected_model}/ 에 모든 파일이 다 들어가 있어야됨.
# TODO 2. 

app = Flask(__name__) 		
CORS(app)

@app.route("/cdhd/debug", methods=['GET', 'POST'])
def test():
    responses = {
            'video' : '',   # * encoded video to base64
            'status': '', # ! error message
        }
    user_info = request.json
    b64 = Utils.binaryToB64(f"{Constants.DEFAULT_USER_DIRECTORY}/for_debug/debug.mp4")
    responses['video'] = b64
    #responses['video'] = "Hello!"
    responses['status'] = "debug!"
    return responses

@app.route("/cdhd/request_video", methods=['GET', 'POST']) 		
def requestVideo():
    responses = {
                'video' : '',   # * encoded video to base64
                'status': '', # ! error message
            }
    requests = {
            'uuid' : '', # * user uuid
            'voice' : '',   # * encoded voice(.wav) to base64 
            'selected_model' : '', # * selected_model
    }
    
    user_info = request.json
    requests['uuid'] = user_info.get('uuid')
    requests['voice'] = user_info.get('voice')
    requests['selected_model'] = user_info.get('selected_model')

    if requests['selected_model'] == Constants.ADJ: selected_model_str = "ADJ"
    elif requests['selected_model'] == Constants.JJS: selected_model_str = "JJS"
    elif requests['selected_model'] == Constants.KSK: selected_model_str = "KSK"
    elif requests['selected_model'] == Constants.MJH: selected_model_str = "MJH"
    elif requests['selected_model'] == Constants.SJG: selected_model_str = "SJG"
    else: responses['status'] = ErrorMessages.CANT_FIND_MODEL

    logger(f"Request from {requests['uuid']} | Processing {selected_model_str}", level='INFO')

    try:
        if Utils.isExistsUserDirectory(requests['uuid']):
            Utils.rmUserDirectory(requests['uuid'])
            logger(f"Remove user directory. | Request from {requests['uuid']} | Processing {selected_model_str}", level="INFO")

        Utils.createFolder(f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}")
        logger(f"Create user directory. | Request from {requests['uuid']} | Processing {selected_model_str}", level="INFO")
        #print(requests['voice'])
        requests['voice'] = Utils.removeBase64Header(requests['voice'])
        #print(requests['voice'])
        request_voice = Utils.b64ToBinary(requests['voice'])
        
        Utils.saveFileToDirectory(request_voice, f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}/{requests['uuid']}.wav")
        Audio_Utils.convertWavSample(f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}/{requests['uuid']}.wav", f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}/{requests['uuid']}_new.wav")
        
        if Audio_Utils.getWavLength(f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}/{requests['uuid']}.wav") > 60:
            logger(f"{ErrorMessages.TOO_LONG_LENGTH} | Request from {requests['uuid']} | Processing {selected_model_str}", level='ERROR')
            responses['status'] = ErrorMessages.TOO_LONG_LENGTH
            return jsonify(responses)
        
        logger(f"User directory created & file loaded | Request from {requests['uuid']} | Processing {selected_model_str}", level='INFO')

    except:
        logger(f"{ErrorMessages.CANT_WRITE_FILE} | Request from {requests['uuid']} | Processing {selected_model_str}", level='ERROR')
        responses['status'] = ErrorMessages.CANT_WRITE_FILE
        return jsonify(responses)
    try:
        # audio preprocessing
        pkl_path = audio_preprocessing.make_pickle(requests['uuid'], selected_model_str)
        # audio testing
        npz_path = test_audio.test_audio(pkl_path, requests['uuid'], selected_model_str)
        # rendering gaosi
        rendering_gaosi.rendering(requests['uuid'], selected_model_str, npz_path)
        # video testing
        test_video.test_video(requests['uuid'], selected_model_str)
        #concat audio file and video file
        logger(f"Merge audio file & video file | Request from {requests['uuid']} | Processing {selected_model_str}", level='INFO')

        Video_Utils.concatAudioVideo(f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}/{requests['uuid']}.wav", 
                                    f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}/complete_{requests['uuid']}.avi")
                                    
        responses['video'] = Utils.binaryToB64(f"{Constants.DEFAULT_USER_DIRECTORY}/{requests['uuid']}/complete_{requests['uuid']}.mp4")
        
        logger(f"All process is successfully finished. | Request from {requests['uuid']} | Processing {selected_model_str}", level='INFO')
        
        return jsonify(responses)
    except:
        logger(f"{ErrorMessages.MODEL_CANT_EXECUTE} | Request from {requests['uuid']} | Processing {selected_model_str}", level='ERROR')
        responses['status'] = ErrorMessages.MODEL_CANT_EXECUTE
        return jsonify(responses)

if __name__ == "__main__":
    app.run(host='192.168.154.67', port=5000)