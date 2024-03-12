import argparse
import typing as tp
from pathlib import Path

from moisesdb.dataset import MoisesDB
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from data import SAD



parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input-dir',
    type=str,
    required=False,
    default= "../../dataset/moisesdb",
    help="Path to directory with moisesdb dataset"
)
parser.add_argument(
    '-o',
    '--output-dir',
    type=str,
    required=False,
    default= "./files",
    help="Path to directory where output .txt file is saved"
)
parser.add_argument(
    '--subset',
    type=str,
    required=False,
    default='test',
    help="Train/test subset of dataset to process"
)
parser.add_argument(
    '--split',
    type=str,
    required=False,
    default='train',
    help="Train/valid split of train dataset. Used if subset=train"
)
parser.add_argument(
    '--sad-cfg-path',
    type=str,
    required=False,
    default="./conf/sad/default.yaml",
    help="Path to Source Activity Detection config file"
)
parser.add_argument(
    '-t',
    '--targets',
    nargs='+',
    required=False,
    default=["vocals"],
    help="Target source. SAD will save salient fragments of vocal audio."
)
args = parser.parse_args()

def main(
        db_dir: str,
        save_dir: str,
        subset: str,
        split: tp.Optional[str],
        targets: tp.List[str],
        sad_cfg_path: DictConfig
) -> None:
    # initialize moisesdb parser
    db = MoisesDB(
        data_path=db_dir,
        sample_rate=44100
    )

    for track in db:
        print(track.id)

        filepath_template = str(Path(track.path) / "{}.wav")

        mixture = torch.from_numpy(track.audio)
        vocals = torch.from_numpy(track.stems['vocals'])
        drums = torch.from_numpy(track.stems['drums'])
        bass = torch.from_numpy(track.stems['bass'])

        length = mixture.size(1)
        temp = torch.zeros(mixture.size())

        torchaudio.save(uri=filepath_template.format('mixture'), src=mixture, sample_rate=44100, format='wav')
        torchaudio.save(uri=filepath_template.format('vocals'), src=vocals, sample_rate=44100, format='wav')
        torchaudio.save(uri=filepath_template.format('drums'), src=drums, sample_rate=44100, format='wav')
        torchaudio.save(uri=filepath_template.format('bass'), src=bass, sample_rate=44100, format='wav')

        # exception
        keys_to_remove = ['vocals', 'bass', 'drums']
        other_dict = {key: value for key, value in track.stems.items() if key not in keys_to_remove}
        temp_array = np.zeros(mixture.size())
        for arr in other_dict.values():
            adjusted_arr = adjust_array(arr, length)
            temp_array += adjusted_arr
        other = torch.from_numpy(temp_array)
        torchaudio.save(uri=filepath_template.format('other'), src=other, sample_rate=44100, format='wav')

        print("saved!")

def adjust_array(array, target_length):
    current_length = array.shape[1]
    if current_length > target_length:
        # if the length of data is exceeded, trim the data.
        return array[:, :target_length]
    elif current_length < target_length:
        # if the length of data is insufficient, pad the data.
        padding = np.zeros((array.shape[0], target_length - current_length))
        return np.hstack((array, padding))
    else:
        return array

if __name__ == '__main__':
    main(
        args.input_dir,
        args.output_dir,
        args.subset,
        args.split,
        args.targets,
        args.sad_cfg_path,
    )

