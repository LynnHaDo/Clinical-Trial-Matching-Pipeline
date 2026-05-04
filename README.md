# Clinical Trial Matching

## Installation guide

### Install clinical trial data:

1. Download data dump from https://aact.ctti-clinicaltrials.org/downloads
2. In the Terminal, set up Docker container:

* Pull Postgres image: `docker pull postgres:16`
* Copy the dump downloaded earlier to `/tmp` in container: `docker cp path_to_dump/postgres.dmp ctdm-db:/tmp/postgres.dmp`
* Populate the data: `docker exec -it ctdm-db pg_restore -U postgres -e -v -O -x -d aact --no-owner /tmp/postgres.dmp`
* (If needed) Drop schema: `docker exec -it ctdm-db psql -U postgres -d aact -c "DROP SCHEMA ctgov CASCADE;"`

3. In pgAdmin (or other UI to view Postgres db):

* View the top 100 rows:

```
SELECT * FROM ctgov.eligibilities
ORDER BY id ASC LIMIT 100
```

* Connect with database:

```postgres://[username]:[password]@db:5432/aact```

* Store the URL as `DATABASE_URL` in `.env`

### Install MIMIC IV patient notes

We obtained the copy of [MIMIC-IV-Note: Deidentified free-text clinical notes](https://physionet.org/content/mimic-iv-note/2.2/) from PhysioNet. Once you have access to data, make a copy of `discharge.csv` to `medical-transcription-processor/datasets`. Then run:

```
python3 load_mimic_to_postgres.py
```

### Install Python Virtual Environment for dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Training

### Clinical Trials BIO Tagging

1. Train the model and view results:

```
python3 clinical_trial_processor.encoder
```

A new model will be saved in either `models`, directory `ncbi_disease` or `tner_bc5cdr` based on your default database (can set in `constants.py`), along with the losses and evaluation.

2. Update the database with extracted graph info:

```
python3 -m clinical_trial_processor.process
```

3. Embed non-boolean edges using ClinicalBERT

```
python3 -m clinical_trial_processor.embed
```

## Run the matching algorithm

```
python3 main.py [some_test_json_file]
```