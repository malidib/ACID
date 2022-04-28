# Welcome to ACID

The Astrophysical CIrcles Detector (ACID) is a computer vision package trained to detect quasi-circular objects in, non-exclusively, astrophysical images. Such objects include impact craters on any well imaged solar system object, eddies (cyclones) on gas planets, boulders on asteroids and comets, and even HI holes in galaxies. 

### Characteristics 
- It is an ensemble model built on top of the MaskRCNN semantic segmentation framework (Matterport implementation). 
- Easy to install and use. No prior experience with machine learning is needed. Can be used on a personal computer.
- Trained on a massive and highly augmented craters-only dataset. Detection of any other object is a form of transfer-learning.
- Returns the location, size, and shape of the detected objects. 
- Packs many convenience functions to preprocess images and postprocess results. 
- Still in active development. 

### Installation & Usage
ACID can be downloaded and used right away. No installation is needed. It does however have many dependencies that must be installed. A conda environment with all of the needed packages can be installed using the provided yml file : conda env create -f acid_env.yml 

Multiple usage examples are included as jupyter notebooks and python scripts. Users are advised to start with example_moon as it is the most detailed. 

### Acknowledgments
A dedicated ACID paper will be published soon. In the meanwhile, if you find ACID useful, please cite this Github repository and the following two papers:

Ali-Dib, M. et al. (2020) Icarus, Volume 345, article id. 113749.

Silburt, A., Ali-Dib, M. et al. (2019) Icarus, Volume 317, p. 27-38.

### License
The MIT License (MIT)

