#!/bin/bash

# Physionet credentials
username=$1
password=$2

# Root project directory
cd /datadrive/

# Physionet resources to save to root project directory
physionet_resources=(
  # MIMIC-III
  "https://physionet.org/files/mimiciii/1.4/"
  # MIMIC-IV and notes
  "https://physionet.org/content/mimiciv/3.0/"
  "https://physionet.org/content/mimic-iv-note/2.2/"
  # EHRNoteQA
  "https://physionet.org/content/ehr-notes-qa-llms/1.0.1/"
  # Hallucination annotations/rewriting
  "https://physionet.org/content/ann-pt-summ/1.0.0/"
  # Medication extraction
  "https://physionet.org/content/medication-labels-mimic-note/1.0.0/"
  # CLIP
  "https://physionet.org/content/mimic-iii-clinical-action/1.0.0/"
  # DiSCQ
  "https://physionet.org/content/discq/1.0/"
  # DischargeMe
  "https://physionet.org/content/discharge-me/1.3/"
  # Phenotype
  "https://physionet.org/content/phenotype-annotations-mimic/1.20.03/"
)

# After cloning, follow instructions in each repository for respective setup
github_repos=(
  # SBDH
  "https://github.com/hibaahsan/MIMIC-SBDH.git"
  # MDACE
  "https://github.com/3mcloud/MDACE.git"
  # LCD Benchmark
  "https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc.git"
)

for url in ${physionet_resources[@]}; do
  echo "${url}"
  sudo wget -r -N -c -np --user "${username}" --password "${password}" "${url}"
done

for url in ${github_repos[@]}; do
  echo "${url}"
  git clone "${url}"
done
