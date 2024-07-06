"""
Contain the base class with basic feature, there will be children classes
"""
from pathlib import Path
from os import PathLike
from typing import Union

class DataProcessorGeneric:
    def __init__(self, raw_data_src:PathLike[Union[Path, str]], mid_processed_src: PathLike[Union[Path, str]], output_data_src: PathLike[Union[Path, str]]) -> None:
        self.raw_data_path = Path(raw_data_src)
        self.mid_process_path = Path(mid_processed_src)
        self.output_data_path = Path(output_data_src)

    def process_all_seed():
        NotImplemented
    
    def process_households_seed():
        NotImplemented

    def process_persons_seed():
        NotImplemented

    def process_all_census():
        NotImplemented

    def process_households_census():
        NotImplemented

    def process_persons_census():
        NotImplemented
        