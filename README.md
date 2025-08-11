# Investigating Privacy Concerns and Mitigations for Healthcare Language and Foundation Models
## NHS England Data Science Team - PhD Internship Project

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

This repository holds code for the work continuing the "Investigating Privacy Concerns and Mitigations for Healthcare Language and Foundation Models" project ([priv-lm-health](https://github.com/nhsengland/priv-lm-health)).

⚠️ This repository is experimental and thus models generated and attacked using this repository are not suitable to deploy into a production environment without further testing and evaluation. ⚠️

This work was conducted as part of an NHS England Data Science PhD Internship project by [Jenny Chim](https://github.com/j-chim) between July and December 2024.

[Link to original project proposal.](https://nhsx.github.io/nhsx-internship-projects/language-foundation-privacy-concern-mitigation/)

_**Note:** Only public or fake data are shared in this repository._

### Project Stucture
- This repository contains code to:
    - Construct the instruction-tuning dataset (`data_processing/`)
    - Run memorisation experiments (`memorisation/`)
    - Run experiments to assess privacy in clinical documentation (`privacy_in_context/`)
    - (see Usage below for more information)
- The accompanying [report](./reports/report.pdf) is also available in the `reports` folder
- More information about the code usage can be found in each sub-directory.

### Getting Started

#### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone git@github.com:nhsengland/pvt_p71_privLMextended.git`

Each sub-directory has its own packages, detailed in a requirements file. To create a suitable environment, change into the sub-directory of interest, then run:
- ```python -m venv <env_name>```
- `source <env_name>/bin/activate`
- `pip install -r requirements.txt`

While part of the model training code shows experiments with larger models (e.g. `meta-llama/Llama-3.1-70B`), the code base is designed to work with compact models as well. Substitute the model names with an alternative hosted on the Hugging Face hub, e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`. 

### Usage
Refer to sub-directories for work package specific instructions.

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more about the [NHS England Data Science](https://nhsengland.github.io/datascience/) visit our [project website](https://nhsengland.github.io/datascience/our_work/) or get in touch at [datascience@nhs.net](mailto:datascience@nhs.net).

<!-- ### Acknowledgements -->

