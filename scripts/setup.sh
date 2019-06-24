#GAN-erator

#change input_text to fit needs
cd /home/ubuntu/attnGAN 
python2 gen_art.py --gpu 0 --input_text "the bear is eating a banana" --data_dir data/coco --model_path models/coco_AttnGAN2.pth --textencoder_path DAMSMencoders/coco/text_encoder100.pth --output_dir output

cd output
mv 0_s_0_g2.png /home/ubuntu/neural-style-docker/content.png

#SCP

scp -i dhananjay-IAM-keypair.pem ubuntu@ec2-54-191-143-121.us-west-2.compute.amazonaws.com:/home/ubuntu/attnGAN/output/* ~/Desktop


#STYLE CONTENT

#DOWNLOAD FILES + RENAME THEM FOR STYLE CONTENT, needs to be png right now

wget https://images.vexels.com/media/users/3/152605/isolated/preview/63b9fe76213c6943365c1ba012064948-rose-head-vintage-tattoo-by-vexels.png
mv 69sfs0D.png content.png
mv 63b9fe76213c6943365c1ba012064948-rose-head-vintage-tattoo-by-vexels.png style.png

#enter the folder, run the docker file on style and content 
#download the file via scp to desktop
cd /home/ubuntu/neural-style-docker
sudo nvidia-docker run --rm -v /home/ubuntu/neural-style-docker:/images albarji/neural-style --content content.png --style style.png

#SCP
scp -i dhananjay-IAM-keypair.pem ubuntu@ec2-54-191-143-121.us-west-2.compute.amazonaws.com:/home/ubuntu/neural-style-docker/content_style_gatys_ss1.0_sw5.0.png ~/Desktop

#to push files from desktop (styles)
scp -i dhananjay-IAM-keypair.pem birds ec2-34-221-90-216.us-west-2.compute.amazonaws.com:/home/ubuntu/neural-style-docker


#SCP from desktop to AWS

scp -i dhananjay-IAM-keypair.pem birds ec2-34-221-90-216.us-west-2.compute.amazonaws.com:/home/ubuntu/

#HOW TO DOWNLOAD GOOGLE FILES

#guide on how to: https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html
#use this to download files from google drive because google drive sucks

gdrivedl https://drive.google.com/file/d/1lqNG75suOuR_8gjoEPYNp8VyT_ufPPig/view

#HOW TO MOVE FILES UP A FOLDER DIRECTORY
#be one directory up and run code
mv my_folder/* .





#MISC

#If things get full go to tmp > var and find the file that is taking all the space on your storage


#to reinstall NVIDIA Driver for Ubuntu 16.04
wget http://us.download.nvidia.com/tesla/410.104/NVIDIA-Linux-x86_64-410.104.run
sudo /bin/sh ./NVIDIA-Linux-x86_64*.run


#ATTNGAN - main repo (not able to get this working yet)
#edit the cfg file to point at the correct directory
vim ~/AttnGAN/code/cfg/DAMSM/bird.yml

#to run the code
cd ~/AttnGAN/code
python pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml --gpu 0

#how to unzip tar files
tar zxvf CUB_200_2011.tgz

