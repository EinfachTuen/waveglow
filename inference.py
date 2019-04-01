# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser


def main(inputMels, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength):
    mel_files = inputMels
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        waveglow.half()
        for k in waveglow.convinv:
            k.float()

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    for i, file_path in enumerate(mel_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print("file_name is ", file_name)
        mel = torch.load(file_path)
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}_synthesis.wav".format(file_name))
        write(audio_path, sampling_rate, audio)
        print(audio_path)


def newest(path):
        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        full_path = max(paths, key=os.path.getctime)

        return full_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--mel_path", default="mels16k")
    parser.add_argument("-s", "--sigma", default=0.6, type=float)
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.1, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    checkpoint_path = newest("./checkpoints")
    print("loaded model is "+checkpoint_path)
    splitCheckpointPath = "".join((map(str, checkpoint_path.split("_")[1:])))
    splitCheckpointPath = splitCheckpointPath.replace("/", "")
    output_dir = "./output_" + str(splitCheckpointPath)
    print("output_dir is ", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mel_folder_files= os.listdir(args.mel_path)
    input_mels = []
    for file in mel_folder_files:
        if ".pt" in file:
            input_mels.append(os.path.join(args.mel_path, file))
        if ".npy" in file:
            input_mels.append(os.path.join(args.mel_path, file))

    print(input_mels)
    main(input_mels, checkpoint_path, args.sigma, output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength)
