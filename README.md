![](https://i.imgur.com/43kusrF.png)

# Art GAN-erator
A project developed over three weeks at the Insight Fellowship that makes use of GANs + Style transfer to lower overhead costs of the ideation process for artists.



## Features
- **Attentional GAN** : Implemented an attentional GAN which is capable of paying attention to the relevant words in a given text input and uses it to generate novel images
- **Neural Style Transfer** : Implemented a fast neural style transfer docker container which takes an input of a content image generated by the AttnGAN and a user given style image and applies the style. 
- **Flask and Docker** : Made use of Flask and Docker to serve the two models in a distributed fashion so that multiple users can use the web application at the same time
- **AWS and PyTorch** : Coded on a p2.xlarge AWS machine making use of PyTorch to modify the models as required.

## Dependencies
1. pipenv
2. python 2.7 (does not work with Python 3!)
3. A GPU!


## Setup
1. Clone repository.

2. Enter the repository folder through your terminal and run setup.sh using: "sh setup.sh" - which will install all the necessary files, models and data. This process can take some time as there are around 2GB of files required for this project.

3. Create a virtual environment in root folder (i.e. art-gan-erator) using "pipenv shell". After this run "pip install -r requirements.txt" to install all the necessary dependencies of the project. 

## Usage

### To run the AttGAN flask application:

To run the web application use:

    flask run
    
Then vigate to to the IP address http://0.0.0.0:5000/input or if using an AWS server go to http://your-public-ip:5000/input in a browser.

### To run the style transfer:

Result of the text-to-image application are stored in the static folder in root directory. To style the image with a custom style, add an image titled as: "style.png" to the folder.


    sudo nvidia-docker run --rm -v ~/art-gan-erator:/images albarji/neural-style --content static/0_s_0_g2.png --style static/style.png --output output

Different strengths of the style can be applied by adding the option ss as follows:

    --ss 0.75 1 1.25
    
Each number will generate an image with given style strength. Smaller numbers reduce the amount of style applied to the generated image. Experimenting with these numbers I found 0.75-1 to be a good sweet spot for producing good bird-art images. Styled images are stored in the output folder.


## Results

The flask application:

![](GAN-gif-video.gif)

The results are stored in static folder as "0_s_0_g2.png".

Some styled results:

With ss = 1

![](sw1.png)

With ss = 2

![](sw2.png)


## Troubleshooting
To install pyenv and python: 
     
    brew install pyenv
    pyenv install 2.7.10
    pyenv global 2.7.10


If any errors when installing python: 

    xcode-select --install
    xcode-select --reset

## Credits
These papers proved invaluable in understanding the problem space:

https://arxiv.org/abs/1508.06576

https://arxiv.org/abs/1711.10485

This work would not have been possible without the code bases from the following 3 repos:

https://github.com/taoxugit/AttnGAN

https://github.com/jcjohnson/neural-style

https://github.com/albarji/neural-style-docker

