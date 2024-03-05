import argparse
import typing as tp
from pathlib import Path

from moisesdb.dataset import MoisesDB
import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from data import SAD



parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input-dir',
    type=str,
    required=False,
    default= "D://dataset//moisesdb//moisesdb_v0.1",
    help="Path to directory with moisesdb dataset"
)
parser.add_argument(
    '-o',
    '--output-dir',
    type=str,
    required=False,
    default= "D://Project//band-split-rope-transformer//files",
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


def prepare_save_line(
        track_name: str,
        start_indices: torch.Tensor,
        window_size: int
) -> tp.Iterable[str]:
    """
    Creates string in format TRACK_NAME START_INDEX END_INDEX.
    """
    for i in start_indices:
        save_line = f"{track_name}\t{i}\t{i + window_size}\n"
        yield save_line


def run_program(
        file_path: Path,
        target: str,
        db: MoisesDB,
        sad: SAD,
) -> None:
    """
    Saves track's name and fragments indices to provided .txt file.
    """
    with open(file_path, 'w') as wf:
        for track in tqdm(db.tracks):
            track_folder = Path(db.data_path) / track
            vocals_folder_exists = any(folder.name.lower().count("vocals") > 0 for folder in track_folder.iterdir() if folder.is_dir())

            if vocals_folder_exists:
                # in moisesdb, there might be multiple waveforms in a single folder
                target_folder = Path(db.data_path) / track / target
                # load from target folder and sum up all wavefroms
                y = sad.load_and_sum_waveforms(target_folder)
                # find indices of salient segments
                indices = sad.calculate_salient_indices(y)
                # write to file
                for line in prepare_save_line(track, indices, int(sad.window_size)):
                    wf.write(line)

            else:
                print("selected target does not exist")
                # TODO:
                #  1. PICK ANY "STEM" INSIDE THE TRACK FOLDER
                #  2. FIND THE LENGTH OF THAT "STEM"
                #  3. CREATE THE EMPTY TENSOR
                #  4. WRITE TO LINE LIKE ABOVE'S TRUE CASE


    return None


def main(
        db_dir: str,
        save_dir: str,
        subset: str,
        split: tp.Optional[str],
        targets: tp.List[str],
        sad_cfg_path: DictConfig
) -> None:
    # initialize MUSDB parser
    split = None if subset == 'test' else split

    db = MoisesDB(
        data_path='..//..//dataset//moisesdb',
        sample_rate=44100
    )
    # initialize Source Activity Detector
    sad_cfg = OmegaConf.load(sad_cfg_path)
    sad = SAD(**sad_cfg)

    # initialize directories where to save indices
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for target in targets:
        if subset == split == 'train':
            file_path = save_dir / f"{target}_moisesdb_train.txt"
        elif subset == 'train' and split == 'valid':
            file_path = save_dir / f"{target}_moisesdb_valid.txt"
        else:
            file_path = save_dir / f"{target}_moisesdb_test.txt"
        # segment data and save indices to .txt file
        run_program(file_path, target, db, sad)

    return None


if __name__ == '__main__':
    main(
        args.input_dir,
        args.output_dir,
        args.subset,
        args.split,
        args.targets,
        args.sad_cfg_path,
    )

