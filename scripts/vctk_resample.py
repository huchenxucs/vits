import os
import argparse
import subprocess
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str)
args, unk = parser.parse_known_args()

data_root = args.data_root

wav_fns = glob(f"{data_root}/wav48_silence_trimmed/**/*_mic1.flac", recursive=True)
print(f"num of files = {len(wav_fns)}")

for wfn in tqdm(wav_fns):
    tgt_fn = wfn.replace("wav48_silence_trimmed", "wav22050")[:-10] + ".wav"
    os.makedirs(os.path.dirname(tgt_fn), exist_ok=True)
    subprocess.check_call(f"sox {wfn} -r 22050 {tgt_fn}", shell=True)

print("Finish!")
