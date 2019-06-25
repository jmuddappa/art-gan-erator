ttnGAN setup script
apt-get install unzip

#data folder
mkdir data
mkdir DAMSMencoders
mkdir models
cd data
#original file: https://drive.google.com/file/d/1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ/view
wget https://www.dropbox.com/s/65fuog8kuwc7kjh/birds.zip
unzip birds.zip
rm birds.zip
cd birds
unzip text.zip

wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar zxvf CUB_200_2011.tgz
rm CUB_200_2011.tgz

#DAMSMencoders folder
cd ../../DAMSMencoders
#link: https://drive.google.com/file/d/1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V/view
wget https://www.dropbox.com/s/rkdox08y6dpwhv5/bird.zip
unzip bird.zip
rm bird.zip

#models
cd ../models
wget https://www.dropbox.com/s/hgwi0h7qxqwsf9d/bird_AttnGAN2.pth
cd ..


