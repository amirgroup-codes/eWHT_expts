from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


CURR_DIR = Path(__file__).resolve().parent
DATA_DIR = CURR_DIR.parent
RAW_MSA_DIR = DATA_DIR / "raw_msa"


def _count_letters(seq: str) -> int:
	"""Count only alphabetic characters (exclude '.', '-', etc.)."""
	return sum(ch.isalpha() for ch in seq)


def _strip_non_letters(seq: str) -> str:
	"""Return sequence containing only letters for clustering input."""
	return re.sub(r"[^A-Za-z]", "", seq)


def _read_fasta(path: Path) -> list[tuple[str, str]]:
	"""Read FASTA/A2M into list of (header, sequence), preserving order."""
	records: list[tuple[str, str]] = []
	header: str | None = None
	seq_chunks: list[str] = []

	with path.open("r", encoding="utf-8") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			if not line:
				continue
			if line.startswith(">"):
				if header is not None:
					records.append((header, "".join(seq_chunks)))
				header = line
				seq_chunks = []
			else:
				seq_chunks.append(line)

	if header is not None:
		records.append((header, "".join(seq_chunks)))

	return records


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
	with path.open("w", encoding="utf-8") as handle:
		for header, seq in records:
			handle.write(f"{header}\n{seq}\n")


def _remove_duplicate_sequences(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
	"""Keep first instance of each exact sequence string."""
	seen: set[str] = set()
	deduped: list[tuple[str, str]] = []
	for header, seq in records:
		if seq in seen:
			continue
		seen.add(seq)
		deduped.append((header, seq))
	return deduped


def _filter_by_min_letter_fraction(records: list[tuple[str, str]], fraction: float = 0.5) -> list[tuple[str, str]]:
	"""Keep sequences with letter-count >= fraction * WT letter-count."""
	if not records:
		return []

	wt_letters = _count_letters(records[0][1])
	threshold = fraction * wt_letters

	kept: list[tuple[str, str]] = []
	for i, (header, seq) in enumerate(records):
		if i == 0:
			# Always keep WT (first sequence).
			kept.append((header, seq))
			continue
		if _count_letters(seq) >= threshold:
			kept.append((header, seq))
	return kept


def _resolve_mmseqs_bin(mmseqs_bin: Path | None = None) -> Path:
	if mmseqs_bin is not None:
		if mmseqs_bin.exists() and mmseqs_bin.is_file():
			return mmseqs_bin
		raise FileNotFoundError(f"Provided MMseqs2 binary not found: {mmseqs_bin}")

	candidate = shutil.which("mmseqs")
	if candidate:
		return Path(candidate)

	raise FileNotFoundError(
		"Could not find MMseqs2 binary. Install `mmseqs` on PATH or pass "
		"`--mmseqs-bin /path/to/mmseqs`."
	)


def _sample_one_per_cluster(
	cluster_tsv: Path,
	id_to_record: dict[str, tuple[str, str]],
	wt_id: str,
	rng: random.Random,
) -> list[tuple[str, str]]:
	"""
	Parse MMseqs cluster TSV and sample one sequence per cluster.
	Guarantees wildtype (wt_id) is selected in its cluster.
	"""
	input_order = {sid: i for i, sid in enumerate(id_to_record.keys())}

	rep_to_members: dict[str, set[str]] = {}
	with cluster_tsv.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			parts = line.split("\t")
			if len(parts) < 2:
				continue
			rep, member = parts[0], parts[1]
			rep_to_members.setdefault(rep, set()).add(member)
			rep_to_members[rep].add(rep)

	if not rep_to_members:
		# If clustering output is empty for any reason, treat each sequence as singleton cluster.
		return list(id_to_record.values())

	reps_sorted = sorted(rep_to_members.keys(), key=lambda sid: input_order.get(sid, 10**12))

	selected_ids: list[str] = []
	for rep in reps_sorted:
		members = [m for m in rep_to_members[rep] if m in id_to_record]
		if not members:
			continue
		if wt_id in members:
			choice = wt_id
		else:
			choice = rng.choice(members)
		selected_ids.append(choice)

	# WT first, then retain input order for the rest.
	selected_unique = list(dict.fromkeys(selected_ids))
	if wt_id in selected_unique:
		selected_unique.remove(wt_id)
		selected_unique.sort(key=lambda sid: input_order[sid])
		selected_unique = [wt_id] + selected_unique
	else:
		selected_unique.sort(key=lambda sid: input_order[sid])

	return [id_to_record[sid] for sid in selected_unique]


def process_a2m_file(path: Path, rng: random.Random, mmseqs_bin: Path) -> None:
	records = _read_fasta(path)
	if not records:
		return

	records = _remove_duplicate_sequences(records)
	records = _filter_by_min_letter_fraction(records, fraction=0.5)
	if not records:
		return

	# Build MMseqs input using letter-only sequences; keep original aligned records for output.
	id_to_record: dict[str, tuple[str, str]] = {}
	mmseqs_input_records: list[tuple[str, str]] = []
	for idx, (header, aligned_seq) in enumerate(records):
		sid = f"seq{idx:08d}"
		id_to_record[sid] = (header, aligned_seq)
		seq_letters_only = _strip_non_letters(aligned_seq)
		if not seq_letters_only:
			# Skip fully non-letter sequences.
			continue
		mmseqs_input_records.append((f">{sid}", seq_letters_only))

	if not mmseqs_input_records:
		return

	wt_id = "seq00000000"

	with tempfile.TemporaryDirectory(prefix="mmseqs_subsample_") as tmp_dir:
		tmp_path = Path(tmp_dir)
		input_fasta = tmp_path / "input.fasta"
		cluster_prefix = tmp_path / "clusterRes"
		mmseqs_tmp = tmp_path / "tmp"

		_write_fasta(input_fasta, mmseqs_input_records)

		cmd = [
			str(mmseqs_bin),
			"easy-cluster",
			str(input_fasta),
			str(cluster_prefix),
			str(mmseqs_tmp),
			"--min-seq-id",
			"0.95",
			"-c",
			"0.8",
			"--cov-mode",
			"1",
			"--remove-tmp-files",
			"1",
		]

		subprocess.run(cmd, check=True, text=True, capture_output=True)

		cluster_tsv = tmp_path / "clusterRes_cluster.tsv"
		if not cluster_tsv.exists():
			# Fallback to all records if clustering TSV was not produced.
			sampled = records
		else:
			sampled = _sample_one_per_cluster(cluster_tsv, id_to_record, wt_id, rng)

	output_path = CURR_DIR / path.name
	sampled_upper = [(header, seq.upper()) for header, seq in sampled]
	_write_fasta(output_path, sampled_upper)


def main() -> None:
	parser = argparse.ArgumentParser(description="Cluster and subsample raw A2M files with MMseqs2.")
	parser.add_argument(
		"--mmseqs-bin",
		type=Path,
		default=None,
		help="Path to the MMseqs2 executable. Defaults to `mmseqs` on PATH.",
	)
	args = parser.parse_args()

	rng = random.Random()
	mmseqs_bin = _resolve_mmseqs_bin(args.mmseqs_bin)
	a2m_files = sorted(RAW_MSA_DIR.glob("*.a2m"))
	for a2m_file in a2m_files:
		process_a2m_file(a2m_file, rng, mmseqs_bin)


if __name__ == "__main__":
	main()
