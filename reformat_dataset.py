import argparse
import typing as tp
from pathlib import Path

from moisesdb.dataset import MoisesDB
import torch
import torch.nn.functional as F
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

    progression_bar = tqdm(db[:2])
    for track in progression_bar:
        filepath_template = str(Path(track.path) / "{}.wav")

        # save mixture
        progression_bar.set_description("track id: [%s] stems: [%s]" % (track.id, str("mixture")))
        mixture = torch.from_numpy(track.audio)
        torchaudio.save(uri=filepath_template.format("mixture"), src=mixture, sample_rate=44100, format='wav')

        # extract size and length from mixture
        length = mixture.size(1)
        temp = torch.zeros(mixture.size())

        # save vocals, drums, bass
        load_stem(filepath_template, track.stems, 'vocals', temp)
        progression_bar.set_description("tack id: [%s] stems: [%s]" % (track.id, str("vocals")))

        load_stem(filepath_template, track.stems, 'drums', temp)
        progression_bar.set_description("tack id: [%s] stems: [%s]" % (track.id, str("drums")))

        load_stem(filepath_template, track.stems, 'bass', temp)
        progression_bar.set_description("tack id: [%s] stems: [%s]" % (track.id, str("bass")))

        # save other
        keys_to_remove = ['vocals', 'bass', 'drums']
        other_dict = {key: value for key, value in track.stems.items() if key not in keys_to_remove}
        temp_array = np.zeros(mixture.size())
        for arr in other_dict.values():
            adjusted_arr = adjust_array(arr, length)
            temp_array += adjusted_arr
        other = torch.from_numpy(temp_array)
        torchaudio.save(uri=filepath_template.format('other'), src=other, sample_rate=44100, format='wav')
        progression_bar.set_description("tack id: [%s] stems: [%s]" % (track.id, str("other")))

def load_stem(fp, stems, target, temp):
    try:
        data = torch.from_numpy(stems[target])
        if not temp.size() == data.size():
            length_diff = temp.size(1) - data.size(1)
            if length_diff > 0:
                data = F.pad(data, (0, length_diff))
            else:
                data = data[:, :temp.size(1)]
    except KeyError:
        data = temp
    torchaudio.save(uri=fp.format(target), src=data, sample_rate=44100, format='wav')

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
