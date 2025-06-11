import synthpop.zone_synthesizer as zs
from pathlib import Path

def run_ipu(
        hh_marg_file: Path,
        p_marg_file: Path,
        hh_sample_file: Path,
        p_sample_file: Path,
        output_path: Path,
        hh_syn_name: str,
        pp_syn_name: str,
        stats_name: str
    ) -> None:

    hh_marg, p_marg, hh_sample, p_sample, xwalk = zs.load_data(
        hh_marg_file, p_marg_file, hh_sample_file, p_sample_file,
    )

    all_households, all_persons, all_stats = zs.synthesize_all_zones(
        hh_marg, p_marg, hh_sample, p_sample, xwalk
    )

    all_households.to_csv(output_path / hh_syn_name, index=False)
    all_persons.to_csv(output_path / pp_syn_name, index=False)
    all_stats.to_csv(output_path / stats_name, index=False)
