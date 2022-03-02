import numpy as np
import os
import tifffile as tif
from typing import Tuple, List
from argparse import ArgumentParser
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Conv2DTranspose, Cropping2D
from tensorflow.keras.optimizers import Adam, SGD

from segmentation.utils.patch_func import make_valid_patch, convert_patches_into_image
from segmentation.utils.transform_data import TransformData, ModifiedStandardization
from segmentation.cldice_loss.cldice import soft_dice_cldice_loss
from utils.data_conversion import Prob2Pred

class AE:
    '''A U-net whose encoder has been pretrained on an AE (U-net+AE)
    '''
    input_size = 320
    output_size = 256
    image_size = 1024
    margin = int((input_size -output_size)/2)
    grid_size = int(image_size/output_size)
    patches_per_img = grid_size**2

    def __init__(self, model_name: str, resume_epoch: int, final_epoch: int, resume_epoch_rec: int, final_epoch_rec: int,\
         transformer: TransformData = ModifiedStandardization, batch_size: int = 8, cldice_loss: bool = False, lr: float = 1e-5) -> None:
        '''
        model_name: used in the name of the saved weights of the model and its output.
        resume_epoch: the epoch from which the model should continue training, 
                        expects the resumed weight at f'./log/{model_name}/weights/weights_{model_name}_{resume_epoch}.h5'
        final_epoch: the model would stop training at this epoch.
        resume_epoch_rec: the epoch from which the reconstruction model should continue training, 
                        expects the resumed weight at f'./log/{model_name}/weights/recweights_{model_name}_{resume_epoch_rec}.h5'
        final_epoch_rec: the reconstruction model would stop training at this epoch.
        transformer: the preprocessing to apply on input patches.
        batch_size: how many patches there are in every batch.
        cldice_loss: if True, uses cldice_loss to train segmentation model
        lr: learning rate used in training
        '''
        if resume_epoch > final_epoch:
            raise ValueError('resume_epoch should not be larger than final_epoch.')
        if resume_epoch_rec > final_epoch_rec:
            raise ValueError('resume_epoch_rec should not be larger than final_epoch_rec.')
        self.model_name = f'cldice{model_name}' if cldice_loss else model_name
        self.resume_epoch = resume_epoch
        self.final_epoch = final_epoch
        self.batch_size = batch_size
        self.transformer = transformer
        self.final_epoch_rec = final_epoch_rec
        self.resume_epoch_rec = resume_epoch_rec
        self.cldice_loss = cldice_loss
        self.lr = lr
        

    def model_architecture(self):
        
        import tensorflow as tf
        input1 = Input((self.input_size,self.input_size,1))
        conv1 = Conv2D(8, 3, activation='relu', padding='same', name='layer0_rec')(input1)
        conv2 = Conv2D(8, 3, activation='relu', padding='same')(conv1)
        maxpool1 = MaxPooling2D(2)(conv2)

        conv3 = Conv2D(16, 3, activation='relu', padding='same')(maxpool1)
        conv4 = Conv2D(16, 3, activation='relu', padding='same')(conv3)
        maxpool2 = MaxPooling2D(2)(conv4)

        conv5 = Conv2D(32, 3, activation='relu', padding='same')(maxpool2)
        conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv5)
        maxpool3 = MaxPooling2D(2, name='features')(conv6)

        conv7 = Conv2D(64, 3, activation='relu', padding='same')(maxpool3)
        conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
        up1 = Conv2DTranspose(64, 2, strides=2, activation='relu', padding='same')(conv8)

        conv11 = Conv2D(32, 3, activation='relu', padding='same')(up1)
        conv12 = Conv2D(32, 3, activation='relu', padding='same')(conv11)
        up2 = Conv2DTranspose(32, 2, strides=2, activation='relu', padding='same')(conv12)

        conv13 = Conv2D(16, 3, activation='relu', padding='same')(up2)
        conv14 = Conv2D(16, 3, activation='relu', padding='same')(conv13)
        up3 = Conv2DTranspose(16, 2, strides=2, activation='relu', padding='same')(conv14)

        conv15 = Conv2D(8, 3, activation='relu', padding='same')(up3)
        conv16 = Conv2D(8, 3, activation='relu', padding='same')(conv15)
        conv17 = Conv2D(1, 1, activation='linear', padding='same')(conv16)
        out_rec = Cropping2D((self.margin,self.margin))(conv17)
 
        ########################################
        
        conv18 = Conv2D(64, 3, activation='relu', padding='same', name='layer0_seg')(maxpool3)
        conv19 = Conv2D(64, 3, activation='relu', padding='same')(conv18)
        up4 = Conv2DTranspose(64, 2, strides=2, activation='relu', padding='same')(conv19)
        merge1 = concatenate([conv6, up4], -1)	
		
        conv20 = Conv2D(32, 3, activation='relu', padding='same')(merge1)
        conv21 = Conv2D(32, 3, activation='relu', padding='same')(conv20)
        up5 = Conv2DTranspose(32, 2, strides=2, activation='relu', padding='same')(conv21)
        merge2 = concatenate([conv4, up5], -1)
		
        conv22 = Conv2D(16, 3, activation='relu', padding='same')(merge2)
        conv23 = Conv2D(16, 3, activation='relu', padding='same')(conv22)
        up6 = Conv2DTranspose(16, 2, strides=2, activation='relu', padding='same')(conv23)
        merge3 = concatenate([conv2, up6],-1)
	    
        conv24 = Conv2D(8, 3, activation='relu', padding='same')(merge3)
        conv25 = Conv2D(8, 3, activation='relu', padding='same')(conv24)
        conv26 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv25)
        out_prob = Cropping2D((self.margin,self.margin), name='seg')(conv26)
    
        rec_model = Model(input1, out_rec)
        seg_model = Model(input1, out_prob)
        #print(rec_model.summary())
        #print(seg_model.summary())
        return rec_model, seg_model

    def sanity_check(self):
        ''' Just a test to verify resuming training of encoder works as intended.
        '''
        rec_model, seg_model = self.model_architecture()
        for layer in rec_model.layers:
            if len( layer.get_weights() ) > 0:
                w_shape, b_shape = (layer.get_weights()[0].shape, layer.get_weights()[1].shape)
                layer.set_weights([np.ones(w_shape), np.ones(b_shape)])
        rec_model.save_weights('rec_weights.h5')
        rec_model.load_weights('rec_weights.h5') # or seg_model.load_weights('rec_weights.h5', by_name=True)
        w, b = seg_model.get_layer('layer0_rec').get_weights()
        print(w, b)

    def extract_training_images_rec(self, tr_path: str, dev_path: str) -> Tuple[np.ndarray, List[str], List[int]]:
        ''' Gathers the information (i.e. duct, names, # z stacks) for input data and preprocesses duct
        tr_path: path to where labeled training duct images are
        dev_path: path to where unlabled training duct images are
        '''
        names_tr, z_list = [], []
        duct_tr = np.zeros([1,1024,1024])
        for path in [tr_path, dev_path]:
            names_LI = [file.replace('duct_', '') for file in os.listdir(path) if file.startswith('duct')]
            for name_LI in names_LI:
                duct = tif.imread(f'{path}/duct_{name_LI}')
                duct_transformed = self.transformer(duct).transform()
                duct_tr = np.append(duct_tr, duct_transformed, axis=0)
                names_tr.append(name_LI)
                z_list.append(duct.shape[0])
        duct_tr = duct_tr[1:]
        return duct_tr, names_tr, z_list


    def load_training_patches_rec(self, duct: np.ndarray, names_tr: List[str], z_list: List[int])\
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
            for i, m in enumerate(mask):
                duct_gen[i], label_gen[i] = make_valid_patch(duct[m], names[m])
            duct_gen = np.expand_dims(duct_gen, axis=-1)
            label_gen = np.expand_dims(label_gen, axis=-1)
            yield duct_gen, label_gen


    def train_rec_model(self, tr_path, dv_path):
        ''' saves learned reconstruction weights and loss at ./log/{model_name}
        '''
        os.makedirs(f'log/{self.model_name}/weights', exist_ok=True)
        os.makedirs(f'log/{self.model_name}/loss', exist_ok=True)
        rec_model = self.model_architecture()[0]
        rec_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        duct_tr, names_tr, z = self.extract_training_images_rec(tr_path, dv_path)
        num_samples = duct_tr.shape[0] * self.patches_per_img
        steps_tr = np.ceil(num_samples / self.batch_size)
        if self.resume_epoch_rec>0:
            print(f'Resuming reconstruction training on epoch {self.resume_epoch_rec} ...')
            rec_model.load_weights(f'log/{self.model_name}/weights/recweights_{self.model_name}_{self.resume_epoch_rec}.h5')
        else: print('Training reconstruction from scratch ...')
        for epoch in np.arange(self.resume_epoch_rec+1, self.final_epoch_rec+1):
            hist = rec_model.fit_generator(self.load_training_patches_rec(duct_tr,names_tr,z), steps_per_epoch=steps_tr, epochs=1)
            rec_model.save_weights(f'log/{self.model_name}/weights/recweights_{self.model_name}_{epoch}.h5')
            np.save(f'log/{self.model_name}/loss/recloss_{self.model_name}_{epoch}.npy', hist.history['loss'])


    def load_test_patches(self, duct: np.ndarray) -> np.ndarray:
        '''Loads each of the 16 patches on the grid in an order to bind them together in test_rec_model and test_model methods
        '''
        duct_pad = np.pad(duct, ((0,0),(self.margin,self.margin),(self.margin,self.margin)), 'constant', constant_values=0)
        duct_pad = np.expand_dims(duct_pad, axis=-1)
        for z in range(duct.shape[0]):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    yield duct_pad[np.newaxis, z, i*self.output_size:i*self.output_size+self.input_size,\
                                                j*self.output_size:j*self.output_size+self.input_size,:]


    def test_rec_model(self, duct_path, write_path: str = None ,epoch_list: List[int] = None):
        ''' Writes probability maps corresponding to the voxel being part of a tube.
        duct_path: the path to where the ductual images to get predictions from is located.
        write_path: the probabilities are written to ./{write_path}/prob. By default write_path={duct_path}/..
        epoch_list: a list of epochs for which to make probabilities. If not specified, only the final_epoch is used.
        '''
        if not epoch_list: epoch_list = [self.final_epoch_rec]
        if write_path is None: write_path = f'{duct_path}/..'
        os.makedirs(f'{write_path}/rec', exist_ok=True)
        for epoch in epoch_list:
            print(f'Testing reconstruction model on epoch {epoch} ...')
            rec_model = self.model_architecture()[0]
            rec_model.load_weights(f'log/{self.model_name}/weights/recweights_{self.model_name}_{epoch}.h5')
            LI_names = [file.replace('duct_', '') for file in os.listdir(duct_path) if file.startswith('duct')]
            for LI_name in LI_names:
                duct = tif.imread(f'{duct_path}/duct_{LI_name}')
                tran = self.transformer(duct)
                duct_transformed = tran.transform()
                rec_patched = rec_model.predict_generator(self.load_test_patches(duct_transformed),\
                    steps=self.patches_per_img*duct.shape[0]).squeeze()
                rec = convert_patches_into_image(rec_patched)
                rec = (rec*3*tran.std) + tran.mean
                rec = rec.astype('uint16')
                tif.imwrite(f'{write_path}/rec/rec-{self.model_name}-{epoch}_{LI_name}', rec)


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


    def train_model(self, duct_path, label_path):
        ''' Saves learned weights and loss at ./log/{model_name}
        '''
        os.makedirs(f'log/{self.model_name}/weights', exist_ok=True)
        os.makedirs(f'log/{self.model_name}/loss', exist_ok=True)
        rec_model, seg_model = self.model_architecture()
        if self.cldice_loss:
            seg_model.compile(optimizer=Adam(learning_rate=self.lr), loss=soft_dice_cldice_loss(), metrics = ['binary_accuracy'])
        else: seg_model.compile(optimizer=Adam(learning_rate=self.lr, loss='binary_crossentropy', metrics = ['binary_accuracy']))
        duct_tr, label_tr, names_tr, z = self.extract_training_images(duct_path, label_path)
        num_samples = duct_tr.shape[0] * self.patches_per_img
        steps_tr = np.ceil(num_samples / self.batch_size)
        if self.resume_epoch == 0:
            print('Resuming training from reconstruction model ...')
            rec_model.load_weights(f'log/{self.model_name}/weights/recweights_{self.model_name}_{self.final_epoch_rec}.h5')
        elif self.resume_epoch>0:
            print('Resuming training on epoch {self.resume_epoch} ...')
            seg_model.load_weights(f'log/{self.model_name}/weights/weights_{self.model_name}_{self.resume_epoch}.h5')
        for epoch in np.arange(self.resume_epoch+1, self.final_epoch+1):
            hist = seg_model.fit_generator(self.load_training_patches(duct_tr,label_tr,names_tr,z), steps_per_epoch=steps_tr, epochs=1)
            seg_model.save_weights(f'log/{self.model_name}/weights/weights_{self.model_name}_{epoch}.h5')
            np.save(f'log/{self.model_name}/loss/segloss_{self.model_name}_{epoch}.npy', hist.history['loss'])


    def test_model(self, duct_path: str, pred_thr: float, write_path: str = None,
                     epoch_list: List[int] = None, make_prob: bool = False) -> None:
        ''' Writes probability maps corresponding to the voxel being part of a tube.
        duct_path: the path to where the ductual images to get predictions from is located.
        write_path: the probabilities are written to ./{write_path}/prob. By default write_path={duct_path}/..
        epoch_list: a list of epochs for which to make probabilities. If not specified, only the final_epoch is used.
        '''
        if epoch_list is None: epoch_list = [self.final_epoch]
        if write_path is None: write_path = f'{duct_path}/..'
        LI_names = [file.replace('duct_', '') for file in os.listdir(duct_path) if file.startswith('duct')]
        for epoch in epoch_list:
            print(f'Testing on epoch {epoch} ...')
            seg_model = self.model_architecture()[1]
            seg_model.load_weights(f'log/{self.model_name}/weights/weights_{self.model_name}_{epoch}.h5')
            for LI_name in LI_names:
                duct = tif.imread(f'{duct_path}/duct_{LI_name}')
                duct_transformed = self.transformer(duct).transform()
                prob_patched = seg_model.predict_generator(self.load_test_patches(duct_transformed),\
                    steps=self.patches_per_img*duct.shape[0]).squeeze()
                prob = convert_patches_into_image(prob_patched)
                prob_name = f'prob-{self.model_name}-{epoch}_{LI_name}'
                Prob2Pred(prob_name, prob, pred_thr).write(f'{write_path}/pred')
                if make_prob:
                    os.makedirs(f'{write_path}/prob', exist_ok=True)
                    tif.imwrite(f'{write_path}/prob/{prob_name}', prob)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tr_duct_path', type=str, default=None)
    parser.add_argument('--dev_duct_path', type=str, default=None)
    parser.add_argument('--tr_label_path', type=str, default=None)
    parser.add_argument('--ts_duct_path', type=str, default=None)
    parser.add_argument('--pred_thr', type=float, default=0.5)
    parser.add_argument('--model_name', type=str, default='ae')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--final_epoch', type=int, default=200)
    parser.add_argument('--resume_epoch_rec', type=int, default=0)
    parser.add_argument('--final_epoch_rec', type=int, default=20)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--train_rec', action='store_true', default=False)
    parser.add_argument('--make_rec', action='store_true', default=False)
    parser.add_argument('--make_prob', action='store_true', default=False)
    parser.add_argument('--cldice_loss', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()
    ae = AE(args.model_name, args.resume_epoch, args.final_epoch, args.resume_epoch_rec, args.final_epoch_rec,\
         cldice_loss=args.cldice_loss, lr=args.lr)
    if args.train_rec: ae.train_rec_model(args.tr_duct_path, args.dev_duct_path)
    if args.train: ae.train_model(args.tr_duct_path, args.tr_label_path)
    if args.make_rec: ae.test_rec_model(args.ts_duct_path)
    if args.ts_duct_path is not None:
        ae.test_model(args.ts_duct_path, args.pred_thr, make_prob=args.make_prob)