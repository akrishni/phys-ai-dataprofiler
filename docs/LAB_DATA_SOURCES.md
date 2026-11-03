# Real-world lab run data: public sources

Compiled 2026-10-20. Verified by fetching each source's own page unless marked (memory).
Ordered by how closely the data resembles an RL trajectory: design choices + conditions + measured outcome,
ideally from a closed-loop campaign.

## Tier A: closed-loop or HTE campaigns (design -> outcome, with batches/plates)

| Source | What it is | Design vars | Outcome | Format / size | Access | License |
|---|---|---|---|---|---|---|
| Attia et al. 2020, fast-charging optimisation (Stanford/MIT/TRI) | A real closed-loop Bayesian-optimisation campaign over battery charging protocols with early-outcome prediction; raw cycling time series. 4 closed-loop rounds of 46-48 cells plus a 46-cell validation batch | charging protocol (CC1_CC2_CC3_CC4 string) | cycle life, per-cycle summary (IR, QCharge, QDischarge, Tavg/Tmax/Tmin, chargetime), per-cycle curves | 5 MATLAB v7.3 struct files (3.7 GB) + 236 per-channel Arbin CSVs (11.3 GB) + JSON zips (6.9 GB); 22.1 GB total | **Hub mirror: huggingface.co/datasets/bsebench-org/tri-attia-fast-charge-2020-raw** (the data.matr.io API no longer answers as of 2026-10-20); code at github.com/chueh-ermon/battery-fast-charging-optimization | CC-BY-4.0 |
| Severson et al. 2019 battery dataset | Predecessor: 124 cells cycled to failure under 72 charging policies, full voltage/current/temperature per cycle | charging policy | cycle life | 3 MATLAB v7.3 batch files, 8.3 GB; also a 2.6 GB Parquet harmonisation (one file per cell) | Hub mirrors: bsebench-org/severson-2019-raw and bsebench-org/severson-2019 | CC-BY-4.0 |
| Buchwald-Hartwig HTE (Ahneman, Doyle, Science 2018) | 3,955 reactions on 1536-well plates: ligand x additive x base x aryl halide | 4 categorical factors | yield | Excel in github.com/rxn4chemistry/rxn_yields `/data` (memory for path) | GitHub | see repo |
| Suzuki-Miyaura HTE (Perera et al., Science 2018) | 5,760 nanomole-scale flow reactions | reactants, ligand, base, solvent | yield | same repo | GitHub | see repo |
| OCx24, Meta FAIR + U Toronto (2024) | Experimental electrocatalysis under industrial conditions: synthesis, XRF, XRD, electrochemical HER/CO2RR, paired with a 1.5 GB computational CSV | composition, synthesis route | activity, selectivity | CSV; experimental summaries in GitHub | github.com/facebookresearch/fairchem (docs/catalysts/datasets/ocx24) | CC-BY-4.0 |
| Adaptyv Bio EGFR design competition (rounds 1 and 2) | Designed protein sequences expressed cell-free and tested by BLI; raw kinetic curves included | sequence | expression, KD, kon/koff | xlsx/CSV plus raw zip | github.com/adaptyvbio/egfr_competition_1 | ODbL (data), Apache 2 (code) |
| Ginkgo Datapoints (Hugging Face `ginkgo-datapoints`) | GDPa1: 242 antibodies x 10 developability assays. GDPx1/x2/x4: DRUG-seq perturbation screens, x4 is 406 compounds x 6 doses, 10M+ rows Parquet. GDPx3: Cell Painting images | compound, dose, cell type, sequence | transcriptomic readout, biophysical assays | CSV / Parquet / images | huggingface.co/ginkgo-datapoints | CC-BY-4.0 on GDPx4; others unspecified |
| Olympus (Aspuru-Guzik group) | 33 experimentally derived benchmark datasets from self-driving-lab and HTE campaigns, used to train emulators for optimiser benchmarking | campaign specific | campaign specific | bundled in the Python package | github.com/aspuru-guzik-group/olympus | MIT |

Note: the `bsebench-org` Hub organisation mirrors about 50 battery cycling datasets (NASA, CALCE, HNEI, TUM, RWTH, Stanford, Oxford and others) with SHA-256 manifests; see bsebench-org/battery-dataset-catalog.

## Tier B: large outcome databases (design partially recorded, no loop)

| Source | What it is | Format / size | Access | License |
|---|---|---|---|---|
| Open Reaction Database | Single-step organic reactions with inputs, conditions, procedure, outcomes, provenance; protobuf schema, JSON/pbtxt renderings; Hugging Face mirror exists | pb.gz via Git LFS, Figshare snapshots | github.com/Open-Reaction-Database/ord-data | CC-BY-SA |
| PubChem BioAssay | Millions of assay results incl. qHTS dose-response from NCATS; several Parquet mirrors on the Hub (e.g. phanvancongthanh/pubchem_bioassay, 100M+ rows) | Parquet / FTP | pubchem.ncbi.nlm.nih.gov, huggingface.co | public domain (PubChem) |
| Materials Data Facility | 880+ datasets, 600+ TB, experimental and computational; Foundry Python client; search for autonomous or combinatorial sets | varies | materialsdatafacility.org, Globus | per dataset |
| Battery Archive (Sandia) | Cycling and abuse-test data from several institutions; web viewer, full CSV by email request | CSV | batteryarchive.org | not stated |
| NREL HTEM DB | Combinatorial sputtered thin-film libraries with synthesis conditions, XRF, XRD, optical and electrical data, REST API (memory; host unreachable from this network today) | JSON API | htem.nrel.gov | not verified |

## Tier C: autonomous-lab systems whose data is not (yet) published as a bulk download

- A-Lab (Berkeley, Ceder group): AlabOS workflow manager on MongoDB is open source (github.com/CederGroupHub/alab_management); the Nature 2023 synthesis run data was not found as a standalone download in this check. Worth asking the group directly.
- Acceleration Consortium (Toronto): REST API contracts and lab control repos are public, no bulk run data found.
- Zenodo and Figshare keyword searches for "self-driving laboratory" returned software and papers, not run data, in the first page of results.

## Formats to expect from instruments (for an episode container)

Plate readers: CSV/XLSX per plate. Liquid handlers (Opentrons, Hamilton): JSON or text run logs. Chromatography:
vendor binary, exported CSV or AnIML. Mass spec: mzML (XML). Spectroscopy: JCAMP-DX, CSV. Microscopy: OME-TIFF,
Zarr. Battery cyclers: vendor CSV or MATLAB. Synchrotron/beamline: HDF5/NeXus, Bluesky documents.
