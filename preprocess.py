import argparse
from tqdm import tqdm
import joblib
import text
from utils import load_filepaths_and_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--njobs", default=16, type=int)
    parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt",
                                                           "filelists/ljs_audio_text_test_filelist.txt"])
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)
        # speed up from 1h30min to 50s
        res = joblib.Parallel(n_jobs=args.njobs)(joblib.delayed(text._clean_text)
                                                 (it[args.text_index], args.text_cleaners)
                                                 for it in tqdm(filepaths_and_text))
        for i in range(len(filepaths_and_text)):
            filepaths_and_text[i][args.text_index] = res[i]

        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])
