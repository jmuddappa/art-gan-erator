from flask import Flask, request, send_from_directory
from flask import request
import os
#from .attnGAN.gen_art import gen_example_from_text
#from user import about
from attnGAN.gen_art import *

project_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_dir, 'images')

app = Flask(__name__, static_url_path='')

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/test")

def run(input_text = "the red bird", gpu_id=0, data_dir="attnGAN/data/birds",
	model_path = "attnGAN/models/bird_AttnGAN2.pth",
	textencoder_path="attnGAN/DAMSMencoders/bird/text_encoder200.pth",
	output_directory="output"):

	phrase = request.args.get('input', default = 'a red bird', type = str)
        cfg.GPU_ID = gpu_id
        cfg.DATA_DIR = data_dir
        cfg.TRAIN.NET_G = model_path
        cfg.TRAIN.NET_E = textencoder_path
	manualSeed = 100
    	random.seed(manualSeed)
    	np.random.seed(manualSeed)
    	torch.manual_seed(manualSeed)
    	if cfg.CUDA:
        	torch.cuda.manual_seed_all(manualSeed)
	now = datetime.datetime.now(dateutil.tz.tzlocal())
    	timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    	output_dir = './output/%s_%s_%s' % \
        	(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
	split_dir, bshuffle = 'train', True
    	image_transform = transforms.Compose([
        	transforms.Scale(int(imsize * 76 / 64)),
        	transforms.RandomCrop(imsize),
        	transforms.RandomHorizontalFlip()])
    	dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    	assert dataset
    	dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
    	algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
    	start_t = time.time()
    	if cfg.TRAIN.FLAG:
        	algo.train()
    	else:
        	'''
        	Generate images from pre-extracted embeddings
        	'''
        	if cfg.B_VALIDATION:
            	# generate images for the whole valid dataset
            		algo.sampling(split_dir)
        	else:
            	# generate images for customized captions
            		gen_example_from_text(
                	input_text, output_directory, dataset.wordtoix, algo)
if __name__ == '__main__':
    app.run(debug=True)
