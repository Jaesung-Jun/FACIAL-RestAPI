import os
import numpy as np
import torch
import pickle
from .model import TfaceGAN
import glob
from constants import Constants

def test_audio(audiopath, uid, selected_model_str):
    #audiopath = f'/home/dilab05/work_directory/capstone/{folder}/examples/audio_preprocessed/test1.pkl'
    
    checkpath = f'{Constants.DEFAULT_NEED_FILES}/{selected_model_str}/Gen-10.mdl'
    outpath = f'{Constants.DEFAULT_USER_DIRECTORY}/{uid}/test-result'
    
    num_params = 71
    out_path = outpath

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #audio_list = glob.glob(audiopath)
    #for audio_path in audio_list:
    
    processed_audio = pickle.load(open(audiopath, 'rb'), encoding=' iso-8859-1')
    
    modelgen = TfaceGAN().cuda()

    modelgen.load_state_dict(torch.load(checkpath))
    modelgen.eval()

    processed_audio = torch.Tensor(processed_audio)
    audioname = audiopath.split('/')[-1].replace('.pkl', '')

    faceparams = np.zeros((processed_audio.shape[0], num_params), float)

    frames_out_path = os.path.join(out_path, audioname+'.npz')
    firstpose = torch.zeros([1,num_params],dtype=torch.float32).unsqueeze(0)

    #with torch.no_grad():
    for i in range(0,processed_audio.shape[0]-127, 127):
        audio = processed_audio[i:i+128,:,:].unsqueeze(0).cuda()

        _faceparam = modelgen(audio, firstpose.cuda())

        firstpose = _faceparam[:,127:128,:]
        faceparams[i:i+128,:] = _faceparam[0,:,:].detach().cpu().numpy()

        # last audio sequence
        if i+127 >= processed_audio.shape[0]-127:
            j = processed_audio.shape[0]-128
            audio = processed_audio[j:j+128,:,:].unsqueeze(0).cuda()
            firstpose = _faceparam[:,j-i:j-i+1,:]
            _faceparam = modelgen(audio, firstpose.cuda())
            faceparams[j:j+128,:] = _faceparam[0,:,:].detach().cpu().numpy()
        
    np.savez(frames_out_path, face = faceparams)
    
    torch.cuda.empty_cache()

    return frames_out_path