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

Select the ids of medical transcriptions from mtsamples' table and put it in `transcriptions.txt`. The trials to be selected from are in `trials.txt`.

```
python3 main.py transcriptions.txt
```

Can add optional `k` to select the top k most similar trials

```
python3 main.py transcriptions.txt --k 5
```

Results will be printed out as follows:

```
Clinical Trial Matching Results        
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Patient ID  ┃ Matching Trials (ID, Score) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│     279      │ No eligible trials found    │
│     552      │ No eligible trials found    │
│     2114     │ No eligible trials found    │
│     4387     │ NCT05940506 (Score: 0)      │
│              │ NCT05647564 (Score: 0)      │
│              │ NCT06353321 (Score: 0)      │
│     3788     │ No eligible trials found    │
│     3902     │ NCT04833530 (Score: 0)      │
│              │ NCT04240392 (Score: 0)      │
│              │ NCT03428789 (Score: 0)      │
│     2498     │ No eligible trials found    │
│     3444     │ NCT05940506 (Score: 0)      │
│              │ NCT03623425 (Score: 0)      │
│              │ NCT05647564 (Score: 0)      │
│     2726     │ No eligible trials found    │
│     716      │ No eligible trials found    │
│     3136     │ No eligible trials found    │
│     4240     │ NCT03281031 (Score: 0)      │
│              │ NCT05386472 (Score: 0)      │
│              │ NCT06155305 (Score: 0)      │
│     293      │ No eligible trials found    │
│     4447     │ No eligible trials found    │
│     4329     │ No eligible trials found    │
│     2102     │ NCT06155305 (Score: 0)      │
│              │ NCT04240392 (Score: 0)      │
│              │ NCT03428789 (Score: 0)      │
│     2829     │ No eligible trials found    │
│     1333     │ NCT06957795 (Score: 0)      │
│              │ NCT03281031 (Score: 0)      │
│              │ NCT03952533 (Score: 0)      │
│     2161     │ NCT06812923 (Score: 0)      │
│              │ NCT05940506 (Score: 0)      │
│              │ NCT03623425 (Score: 0)      │
│     4805     │ No eligible trials found    │
│     3184     │ NCT05940506 (Score: 0)      │
│              │ NCT03623425 (Score: 0)      │
│              │ NCT05647564 (Score: 0)      │
│     2849     │ No eligible trials found    │
│     289      │ No eligible trials found    │
│     865      │ No eligible trials found    │
│     321      │ NCT03952533 (Score: 3)      │
│              │ NCT05386472 (Score: 0)      │
│              │ NCT06814652 (Score: 0)      │
│     2449     │ NCT03952533 (Score: 7)      │
│              │ NCT04240392 (Score: 1)      │
│              │ NCT06814652 (Score: 0)      │
│     2064     │ No eligible trials found    │
│     584      │ No eligible trials found    │
│     4559     │ NCT05940506 (Score: 0)      │
│              │ NCT03623425 (Score: 0)      │
│              │ NCT06353321 (Score: 0)      │
│     995      │ No eligible trials found    │
│     673      │ NCT06812923 (Score: 0)      │
│              │ NCT05940506 (Score: 0)      │
│              │ NCT03623425 (Score: 0)      │
│     3992     │ NCT05940506 (Score: 0)      │
│              │ NCT03623425 (Score: 0)      │
│              │ NCT04806828 (Score: 0)      │
│     2160     │ NCT06155305 (Score: 0)      │
│              │ NCT04833530 (Score: 0)      │
│              │ NCT07222306 (Score: 0)      │
│     4431     │ No eligible trials found    │
│     4349     │ NCT03952533 (Score: 4)      │
│              │ NCT03281031 (Score: 0)      │
│              │ NCT05386472 (Score: 0)      │
│     854      │ No eligible trials found    │
│     4131     │ NCT07076719 (Score: 0)      │
│     803      │ No eligible trials found    │
│      82      │ NCT06353321 (Score: 0)      │
│     2490     │ No eligible trials found    │
│     874      │ No eligible trials found    │
│     1999     │ No eligible trials found    │
│     1550     │ No eligible trials found    │
│     3046     │ No eligible trials found    │
│     2975     │ No eligible trials found    │
│     3790     │ No eligible trials found    │
│     3159     │ NCT04240392 (Score: 1)      │
│              │ NCT06804343 (Score: 0)      │
│              │ NCT04833530 (Score: 0)      │
│     2717     │ NCT04363892 (Score: 0)      │
│              │ NCT07076719 (Score: 0)      │
│     252      │ NCT03952533 (Score: 0)      │
│              │ NCT07412925 (Score: 0)      │
│              │ NCT06814652 (Score: 0)      │
│     2905     │ NCT05566821 (Score: 0)      │
│              │ NCT03267875 (Score: 0)      │
└──────────────┴─────────────────────────────┘
```