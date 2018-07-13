# Generative Image Inpainting For Chalearn ECCV 2018 Looking at People Challenge*

## Instructions to Run

0. Requirements:
    * Install python3.
    * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 1.3.0, 1.4.0, 1.5.0, 1.6.0, 1.7.0).
    * Install tensorflow toolkit [neuralgym](https://github.com/JiahuiYu/neuralgym) (run `pip install git+https://github.com/JiahuiYu/neuralgym`).
1. Data Preprocessing
	* Challenge dataset comes with masked images and their corresponding masked areas which should be combined to create full images
	* Run `prepare_dataset_2.py` with folder paths (absolute paths) containing original masked images and target directory where images will be saved. (both folders should contain subfolders X/ and Y/) see following lines for example:
	`python prepare_dataset_2.py  --MaskedImageFolder  /raid/users/geunlu/datasets/inpainting/still-masked/train/ --FullImageFolder /raid/users/geunlu/datasets/inpainting/still-nonmasked/train/`
	* This step should be repeated for training and testing images.
	* The above code will produce original unmasked images under `FullImageFolder/X/` as well as binary masks under `FullImageFolder/Y/`. Also, imagefiles.flist will be created under FullImageFolder/
2. Training with pre-trained model:
    * Prepare training images .flist (produced in preprocessing stage)
    * Modify [inpaint.yml](/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 'release_imagenet_256'.
    * Run `python train.py`.
3. Testing with the newly trained model on challenge dataset:
    * Run `python test_multi.py --image_dir /raid/users/geunlu/datasets/inpainting/still-nonmasked/test/X/ --mask_dir /raid/users/geunlu/datasets/inpainting/still-nonmasked/test/Y/ --output_dir examples/people-chalearn --checkpoint_dir model_logs/20180710102226667005_dgx-server_people_NORMAL_glsgan_gp_model_logs`.
	
	
## inpaint.yml (modifications)

* LOG_DIR: model_logs (this folder contains pretrained models also newly trained models are saved here)
* MODEL_RESTORE: 'release_places2_256' (pretrained model folder under model_logs/)
* DATA_FLIST:
  people: [
  '/raid/users/geunlu/datasets/inpainting/still-nonmasked/train/imagefiles.flist',
  '/raid/users/geunlu/datasets/inpainting/stil-nonmasked/test/imagefiles.flist'
  ]
* GLS_GAMMA: 0.001 # Generalized Loss Sensitive GAN loss gamma parameter
* GLS_SLOPE: 0.3	 # Generalized Loss Sensitive GAN loss slope parameter

## Testing Durations
* Testing will take ~12 hours for 6160 images. The reason for this is the network used in this project can only make predictions for a fixed sized image HxW, but the challege dataset contains images with different sizes. Therefore in test_multi.py the network is initialized for each testing image from scratch.

