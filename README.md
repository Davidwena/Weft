# Weft

This is the source repository for **Weft: Secure Selective Band-Join Queries for Cross-Bank Anti-Money Laundering Investigations**.

Weft implements dealer-assisted secure comparisons, fixed-length compaction, nested-loop (NL) and dominance-counting (DC) band joins, shared multi-query execution, and batched investigations of multiple suspects.

## Layout

| Directory | Contents |
| --- | --- |
| `include/bj/` | Protocol interfaces, arithmetic, networking, query plans, and operators. |
| `src/` | Bank runtime, comparison primitives, compaction, NL/DC, and query execution. |
| `dealer/` | Parallel preprocessing generation and the isolated dealer executable. |
| `apps/` | Bank entry point, material-layout inspection, and plaintext reference. |
| `bench/` | Executable drivers for primitives, operators, and investigations. |
| `tools/` | Experiment orchestration, synthetic generators, AMLworld conversion, and private network setup. |
| `scripts/` | Key build, dataset preparation, and experiment launchers. |
| `examples/` | A public multi-query configuration. |
| `tests/` | Unit tests, allocation checks, and parallel dealer kernel tests. |

## Build

Use Linux on x86-64 with AES-NI, a C++20 compiler, CMake 3.20+, OpenMP, Python 3.10+, `iproute2`, and `util-linux`. Optional tests also require Boost headers. CMake finds or downloads nlohmann/json and, when tests are enabled, GoogleTest.

```sh
python3 -m pip install -r requirements.txt
bash scripts/build.sh
```

Set `JOBS` to change build parallelism, `PYTHON` to select a Python interpreter, or pass `--build /path/to/build` to an experiment. To build tests, run `bash scripts/build.sh -DWEFT_BUILD_TESTS=ON`, then `ctest --test-dir build --output-on-failure`.

## Experiments

Run these commands from the repository root. Each launcher accepts `--help`. Defaults use three banks and 16 bank threads; the parallel dealer uses 16 threads. `--profile` selects `lan` (10 Gbit/s, 0.2 ms target RTT), `wan` (1 Gbit/s, 20 ms), `low-wan` (100 Mbit/s, 20 ms), or unshaped `local`.

| Experiment | Command |
| --- | --- |
| Comparison primitives | `bash scripts/run_primitives.sh --counts 1024 16384 --widths 32 64` |
| NL versus DC join operators | `bash scripts/run_operators.sh --K 512 --operator both` |
| Compaction | `bash scripts/run_operators.sh --phase compact --N 1048576 --K 512` |
| Complete synthetic investigation | `bash scripts/run_investigation.sh --N 1048576 --K 512 --pattern P1 --operator nl` |
| Large-K DC investigation | `bash scripts/run_investigation.sh --N 1048576 --K 8192 --nu-in 8192 --nu-out 8192 --operator dc` |
| Shared multi-query execution | `bash scripts/run_multi_query.sh --query examples/multi_query.json` |
| Batched suspects | `bash scripts/run_batch.sh --N 1048576 --K 512 --suspects 1 4 8 16` |
| Dealer generation and transfer | `bash scripts/run_dealer.sh --banks 2 3 4 5 6 7 8 --modes prg random --repeats 2` |

`--pattern` selects P1, P2, P3, or P4. Change `--K`, `--N`, `--n`, and `--profile` to vary capacity, input size, banks, and network. The batch launcher uses NL. The multi-query launcher reads operators and public dimensions from its configuration.

The launchers create disposable user/network namespaces and shape only their private links; Linux must allow unprivileged user namespaces. Fresh preprocessing is generated in tmpfs without persistent writes, consumed once, and deleted. The default scratch filesystem is `/dev/shm`; use `--scratch` to select another tmpfs. The `random` mode uses OS CSPRNG-generated explicit shares.

Runs write new output files under `outputs/`, or to a fresh path supplied with `--output`.

## AMLworld

Download [AMLworld / IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), linked by the [official IBM repository](https://github.com/IBM/AML-Data). Extract the matching `*_Trans.csv` and `*_accounts.csv` files into a directory outside this repository.

```sh
bash scripts/prepare_amlworld.sh --source-dir /path/to/AMLworld --variant HI-Medium --output /path/to/medium
bash scripts/prepare_amlworld.sh --source-dir /path/to/AMLworld --variant HI-Large --days 90 --output /path/to/large90
bash scripts/run_investigation.sh --data /path/to/medium --pattern P1 --K 512 --operator nl
bash scripts/run_batch.sh --data /path/to/medium --K 512 --suspects 2 --subjects 1 2
```

Conversion uses the supplied entity mapping, exact cents, US-dollar transactions, and the paper's SHA-256 bank assignment. `--days` selects a window starting at the first observed clean day; `--begin` changes its start. Entity IDs for `--subjects` are listed in the converted dataset's `private/entity_counts.csv`. Complete investigations check outputs against the plaintext reference; batch investigations check every selected suspect.
