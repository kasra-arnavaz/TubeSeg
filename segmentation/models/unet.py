import tensorflow as tf
import tifffile as tif
import numpy as np
import os
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Conv2DTranspose, Cropping2D
from tensorflow.python.keras.optimizers import Adam, SGD
from typing import Tuple, List
from argparse import ArgumentParser

from segmentation.utils.patch_func import make_valid_patch, convert_patches_into_image
from segmentation.utils.transform_data import TransformData, ModifiedStandardization
from segmentation.cldice_loss.cldice import soft_dice_cldice_loss
from utils.data_conversion import Prob2Pred

class UNet:
    '''U-net segmentation model
    '''

    input_size = 320
    output_size = 256
    image_size = 1024
    margin = int((input_size -output_size)/2)
    grid_size = int(image_size/output_size)
    patches_per_img = grid_size**2

    def __init__(self, model_name: str, resume_epoch: int, final_epoch: int, \
        transformer: TransformData = ModifiedStandardization, batch_size: int = 8, cldice_loss: bool = False, lr: float = 1e-4) -> None:
        '''
        model_name: used in the name of the saved weights of the model and its output.
        resume_epoch: the epoch from which the model should continue training, 
                        expects the resumed weight at f'./log/{model_name}/weights/weights_{model_name}_{resume_epoch}.h5'
        final_epoch: the model would stop training at this epoch.
        transformer: the preprocessing to apply on input patches.
        batch_size: how many patches there are in every batch.
        cldice_loss: if True, uses cldice_loss to train model
        lr: learning rate used in training
        '''
        if resume_epoch > final_epoch:
            raise ValueError('resume_epoch should not be larger than final_epoch.')
        self.model_name = f'cldice{model_name}' if cldice_loss else model_name
        self.resume_epoch = resume_epoch
        self.final_epoch = final_epoch
        self.batch_size = batch_size
        self.transformer = transformer
        self.cldice_loss = cldice_loss
        self.lr = lr

    def model_architecture(self):
        ''' A U-net based binary segmentation architecture.
        '''
        input1 = Input((self.input_size,self.input_size,1))
        conv1 = Conv2D(8, 3, activation='relu', padding='same')(input1)
        conv2 = Conv2D(8, 3, activation='relu', padding='same')(conv1)
        maxpool1 = MaxPooling2D(2)(conv2)

        conv3 = Conv2D(16, 3, activation='relu', padding='same')(maxpool1)
        conv4 = Conv2D(16, 3, activation='relu', padding='same')(conv3)
        maxpool2 = MaxPooling2D(2)(conv4)

        conv5 = Conv2D(32, 3, activation='relu', padding='same')(maxpool2)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv5)
        maxpool3 = MaxPooling2D(2)(conv6)

        conv7 = Conv2D(64, 3, activation='relu', padding='same')(maxpool3)
        conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
        up1 = Conv2DTranspose(64, 2, strides=2, activation='relu', padding='same')(conv8)
        merge1 = concatenate([conv6, up1], -1)	

        conv11 = Conv2D(32, 3, activation='relu', padding='same')(merge1)
        conv12 = Conv2D(32, 3, activation='relu', padding='same')(conv11)
        up2 = Conv2DTranspose(32, 2, strides=2, activation='relu', padding='same')(conv12)
        merge2 = concatenate([conv4, up2], -1)
	
        conv13 = Conv2D(16, 3, activation='relu', padding='same')(merge2)
        conv14 = Conv2D(16, 3, activation='relu', padding='same')(conv13)
        up3 = Conv2DTranspose(16, 2, strides=2, activation='relu', padding='same')(conv14)
        merge3 = concatenate([conv2, up3], -1)

        conv15 = Conv2D(8, 3, activation='relu', padding='same')(merge3)
        conv16 = Conv2D(8, 3, activation='relu', padding='same')(conv15)
        conv17 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv16)
        out_prob = Cropping2D((self.margin,self.margin), name='seg')(conv17)

        model = Model(input1, out_prob)
        #print(model.summary())
        return model 


    def extract_training_images(self, duct_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
        ''' Gathers the information (i.e. duct, label, names, # z stacks) for input data and preprocesses duct
        duct_path: path to where training duct images are
        label_path: path to where training label images are
        '''
        LI_names = [file.replace('duct_', '') for file in os.listdir(duct_path) if file.startswith('duct')]
        z_list = []
        duct_tr, label_tr = np.zeros([1,1024,1024]), np.zeros([1,1024,1024])
        for LI_name in LI_names:
            duct = tif.imread(f'{duct_path}/duct_{LI_name}')
            label = tif.imread(f'{label_path}/label_{LI_name}')
            duct_transformed = self.transformer(duct).transform()
            duct_tr = np.append(duct_tr, duct_transformed, axis=0)
            label_tr = np.append(label_tr, label, axis=0)
            z_list.append(duct.shape[0])
        duct_tr, label_tr = duct_tr[1:], label_tr[1:]
        label_tr = np.clip(label_tr, 0, 1) # because the intensity for foreground images are sometimes 128 or 255.
        return duct_tr, label_tr, LI_names, z_list


    def load_training_patches(self, duct: np.ndarray, label: np.ndarray, names_tr: List[str], z_list: List[int]) \
        -> Tuple[np.ndarray, np.ndarray]:
        ''' Selects patches from images at random while avoiding forbidden regions and rotates them
        '''
        while True:
            names = []
            for i, name in enumerate(names_tr):
                names += [name]*z_list[i]
            num_samples = duct.shape[0]
            duct_gen = np.zeros([self.batch_size, self.input_size, self.input_size])
            label_gen = np.zeros([self.batch_size, self.output_size, self.output_size])
            mask = np.random.randint(num_samples, size=self.batch_size)
            for i,m in enumerate(mask):
                duct_gen[i], label_gen[i] = make_valid_patch(duct[m], names[m], label[m])
                degree = np.random.randint(4)
                duct_gen[i] = np.rot90(duct_gen[i], degree, (0,1))
                label_gen[i] = np.rot90(label_gen[i], degree, (0,1))
            duct_gen = np.expand_dims(duct_gen, axis=-1)
            label_gen = np.expand_dims(label_gen, axis=-1)
            yield duct_gen, label_gen


    def train_model(self, duct_path: str, label_path: str) -> None:
        ''' Saves learned weights and loss at ./log/{model_name}
        '''
        my_model = self.model_architecture()
        if self.cldice_loss:
            my_model.compile(optimizer=Adam(learning_rate=self.lr), loss=soft_dice_cldice_loss(), metrics = ['binary_accuracy'])
        else: my_model.compile(optimizer=Adam(learning_rate=self.lr), loss='binary_crossentropy', metrics = ['binary_accuracy'])
        duct_tr, label_tr, names_tr, z_list = self.extract_training_images(duct_path, label_path)
        num_samples = label_tr.shape[0] * self.patches_per_img
        steps_tr = np.ceil(num_samples / self.batch_size)
        if self.resume_epoch>0:
            print(f'Resuming training on epoch {self.resume_epoch} ...')
            my_model.load_weights(f'log/{self.model_name}/weights/weights_{self.model_name}_{self.resume_epoch}.h5')
        else: print('Training from scratch ...')
        for epoch in np.arange(self.resume_epoch+1, self.final_epoch+1):
            hist = my_model.fit_generator(self.load_training_patches(duct_tr,label_tr,names_tr,z_list), steps_per_epoch=steps_tr, epochs=1)
            os.makedirs(f'log/{self.model_name}/weights', exist_ok=True)
            my_model.save_weights(f'log/{self.model_name}/weights/weights_{self.model_name}_{epoch}.h5')
            os.makedirs(f'log/{self.model_name}/loss', exist_ok=True)
            np.save(f'log/{self.model_name}/loss/segloss_{self.model_name}_{epoch}.npy', hist.history['loss'])

    def load_test_patches(self, duct: np.ndarray) -> np.ndarray:
        '''Loads each of the 16 patches on the grid in an order to bind them together test_model method
        '''
        duct_pad = np.pad(duct, ((0,0),(self.margin,self.margin),(self.margin,self.margin)), 'constant', constant_values=0)
        duct_pad = np.expand_dims(duct_pad, axis=-1)
        for z in range(duct.shape[0]):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    yield duct_pad[np.newaxis, z, i*self.output_size:i*self.output_size+self.input_size,\
                                                j*self.output_size:j*self.output_size+self.input_size,:]
   
    def test_model(self, duct_path: str, pred_thr: float, write_path: str = None,
                    epoch_list: List[int] = None, make_prob: bool = False) -> None:
        ''' Writes probability maps corresponding to the voxel being part of a tube.
        duct_path: the path to where the ductual images to get predictions from is located.
        write_path: the probabilities are written to ./{write_path}/prob. By default write_path={duct_path}/..
        epoch_list: a list of epochs for which to make probabilities. If not specified, only the final_epoch is used.
        '''
        if epoch_list is None: epoch_list = [self.final_epoch]
        if write_path is None: write_path = f'{duct_path}/..'
        os.makedirs(f'{write_path}/prob', exist_ok=True)
        LI_names = [file.replace('duct_', '') for file in os.listdir(duct_path) if file.startswith('duct')]
        for epoch in epoch_list:
            print(f'Testing on epoch {epoch} ...')
            my_model = self.model_architecture()
            my_model.load_weights(f'log/{self.model_name}/weights/weights_{self.model_name}_{epoch}.h5')
            for LI_name in LI_names:
                duct = tif.imread(f'{duct_path}/duct_{LI_name}')
                duct_transformed = self.transformer(duct).transform()
                prob_patched = my_model.predict_generator(self.load_test_patches(duct_transformed), steps=self.patches_per_img*duct.shape[0]).squeeze()
                prob = convert_patches_into_image(prob_patched)
                prob_name = f'prob-{self.model_name}-{epoch}_{LI_name}'
                Prob2Pred(prob_name, prob, pred_thr).write(f'{write_path}/pred')
                if make_prob:
                    tif.imwrite(f'{write_path}/prob/{prob_name}', prob)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tr_duct_path', type=str, default=None)
    parser.add_argument('--tr_label_path', type=str, default=None)
    parser.add_argument('--ts_duct_path', type=str, default=None)
    parser.add_argument('--pred_thr', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='unet')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--final_epoch', type=int, default=200)
    parser.add_argument('--make_prob', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cldice_loss', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    unet = UNet(args.model_name, args.resume_epoch, args.final_epoch, cldice_loss=args.cldice_loss, lr=args.lr)
    if args.train: unet.train_model(args.tr_duct_path, args.tr_label_path)
    if args.ts_duct_path is not None:
        unet.test_model(args.ts_duct_path, args.pred_thr, make_prob=args.make_prob)




