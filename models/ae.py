from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from utils.data_utils import *
from utils.patch_utils import *
from utils.transform_data import *

class AE:

    input_size = 320
    output_size = 256
    image_size = 1024
    margin = int((input_size -output_size)/2)
    grid_size = int(image_size/output_size)
    patches_per_img = grid_size**2

    def __init__(self, name, resume_epoch, final_epoch, resume_epoch_rec, final_epoch_rec, transformer, batch_size=8):
        self.name = name
        self.resume_epoch = resume_epoch
        self.final_epoch = final_epoch
        self.batch_size = batch_size
        self.transformer = transformer
        self.final_epoch_rec = final_epoch_rec
        self.resume_epoch_rec = resume_epoch_rec
        
    

    def model_architecture(self):
        
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
        import tensorflow as tf
        rec_model, seg_model = self.model_architecture()
        for layer in rec_model.layers:
            if len( layer.get_weights() ) > 0:
                w_shape, b_shape = (layer.get_weights()[0].shape, layer.get_weights()[1].shape)
                layer.set_weights([np.ones(w_shape), np.ones(b_shape)])
        rec_model.save_weights('rec_weights.h5')
        rec_model.load_weights('rec_weights.h5') # or seg_model.load_weights('rec_weights.h5', by_name=True)
        w, b = seg_model.get_layer('layer0_rec').get_weights()
        print(w, b)

    def extract_training_images_rec(self, tr_path, dev_path):
        names_tr, z_list = [], []
        x_tr, y_tr = np.zeros([1,1024,1024]), np.zeros([1,1024,1024])
        for path in [tr_path, dev_path]:
            for name_parser, x in TifReader(path):
                x_transformed = self.transformer(x).transform()
                x_tr = np.append(x_tr, x_transformed, axis=0)
                names_tr.append(name_parser.name)
                z_list.append(x.shape[0])
        x_tr = x_tr[1:]
        return x_tr, names_tr, z_list


    def load_training_patches_rec(self, x, names_tr, z_list):
        while True:
            names = []
            for i, name in enumerate(names_tr):
                names += [name]*z_list[i]
            num_samples = x.shape[0]
            x_gen = np.zeros([self.batch_size, self.input_size, self.input_size])
            target_gen = np.zeros([self.batch_size, self.output_size, self.output_size])
            mask = np.random.randint(num_samples, size=self.batch_size)
            if np.any(y==None): y = [None]*num_samples
            for i, m in enumerate(mask):
                x_gen[i], target_gen[i] = make_valid_patch(x[m], names[m])
            x_gen = np.expand_dims(x_gen, axis=-1)
            target_gen = np.expand_dims(target_gen, axis=-1)
            yield x_gen, target_gen


    def train_rec_model(self, tr_path, dv_path):
        
        os.makedirs(f'results/{self.name}/2d/rec/weights', exist_ok=True)
        os.makedirs(f'results/{self.name}/2d/rec/hist', exist_ok=True)
        rec_model = self.model_architecture()[0]
        rec_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')
        x_tr, names_tr, z = self.extract_training_images_rec(tr_path, dv_path)
        num_samples = x_tr.shape[0] * self.patches_per_img
        steps_tr = np.ceil(num_samples / self.batch_size)
        if self.resume_epoch_rec>0:
            print(f'Resuming reconstruction training on epoch {self.resume_epoch_rec} ...')
            rec_model.load_weights(f'results/{self.name}/2d/rec/weights/{self.name}_{self.resume_epoch_rec}_rec_weights.h5')
        else: print('Training reconstruction from scratch ...')
        for epoch in np.arange(self.resume_epoch_rec+1, self.final_epoch_rec+1):
            hist = rec_model.fit_generator(self.load_training_patches_rec(x_tr,names_tr,z), steps_per_epoch=steps_tr, epochs=1)
            rec_model.save_weights(f'results/{self.name}/2d/rec/weights/weights-rec-{self.name}-{epoch}.h5')
            np.save(f'results/{self.name}/2d/rec/hist/{self.name}_{epoch}_rec_loss.npy', hist.history['loss'])


    def load_test_patches(self, x):

        x_pad = np.pad(x, ((0,0),(self.margin,self.margin),(self.margin,self.margin)), 'constant', constant_values=0)
        x_pad = np.expand_dims(x_pad, axis=-1)
        for z in range(x.shape[0]):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    yield x_pad[np.newaxis, z, i*self.output_size:i*self.output_size+self.input_size,\
                                                j*self.output_size:j*self.output_size+self.input_size,:]


    def test_rec_model(self, path, epoch_list=None):

        if not epoch_list: epoch_list = [self.final_epoch_rec]
        for epoch in epoch_list:
            print(f'Testing reconstruction model on epoch {epoch} ...')
            rec_model = self.model_architecture()[0]
            rec_model.load_weights(f'results/{self.name}/2d/rec/weights/weights-rec-{self.name}-{epoch}.h5')
            for name_parser, x in TifReader(path):
                tran = self.transformer(x)
                x_transformed = tran.transform()
                rec_patched = rec_model.predict_generator(self.load_test_patches(x_transformed), steps=self.patches_per_img*x.shape[0]).squeeze()
                rec = convert_patches_into_image(rec_patched)
                rec = (rec*3*tran.std) + tran.mean
                rec = rec.astype('uint16')
                name_parser.category = f'rec-{self.name}-{epoch}'
                TifWriter(f'results/{self.name}/2d/rec/images/rec/{name_parser.split}', name_parser, rec).write()


    def extract_training_images(self, x_path, y_path):

        names_tr, z_list = [], []
        x_tr, y_tr = np.zeros([1,1024,1024]), np.zeros([1,1024,1024])
        for name_parser, x in TifReader(x_path):
            name_parser.category = 'label'
            x_transformed = self.transformer(x).transform()
            y = TifReader(y_path).read(name_parser.name)[1]
            x_tr = np.append(x_tr, x_transformed, axis=0)
            y_tr = np.append(y_tr, y, axis=0)
            names_tr.append(name_parser.name)
            z_list.append(x.shape[0])
        x_tr, y_tr = x_tr[1:], y_tr[1:]
        y_tr = np.clip(y_tr, 0, 1) # because the intensity for foreground images are sometimes 128 or 255.
        return x_tr, y_tr, names_tr, z_list

        
    def load_training_patches(self, x, y, names_tr, z_list):

        while True:
            names = []
            for i, name in enumerate(names_tr):
                names += [name]*z_list[i]
            num_samples = x.shape[0]
            x_gen = np.zeros([self.batch_size, self.input_size, self.input_size])
            target_gen = np.zeros([self.batch_size, self.output_size, self.output_size])
            mask = np.random.randint(num_samples, size=self.batch_size)
            if np.any(y==None): y = [None]*num_samples
            for i, m in enumerate(mask):
                x_gen[i], target_gen[i] = make_valid_patch(x[m], names[m], y[m])
            x_gen = np.expand_dims(x_gen, axis=-1)
            target_gen = np.expand_dims(target_gen, axis=-1)
            yield x_gen, target_gen


    def train_model(self, x_path, y_path):

        os.makedirs(f'results/{self.name}/2d/seg/weights', exist_ok=True)
        os.makedirs(f'results/{self.name}/2d/seg/hist', exist_ok=True)
        rec_model, seg_model = self.model_architecture()
        seg_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['binary_accuracy'])
        x_tr, y_tr, names_tr, z = self.extract_training_images(x_path, y_path)
        num_samples = x_tr.shape[0] * self.patches_per_img
        steps_tr = np.ceil(num_samples / self.batch_size)
        if self.resume_epoch == 0:
            print('Resuming training from reconstruction model ...')
            rec_model.load_weights(f'results/{self.name}/2d/rec/weights/weights-rec-{self.name}-{epoch}.h5')
        elif self.resume_epoch>0:
            print('Resuming training on epoch {self.resume_epoch} ...')
            seg_model.load_weights(f'results/{self.name}/2d/seg/weights/weights-seg-{self.name}-{epoch}.h5')
        for epoch in np.arange(self.resume_epoch+1, self.final_epoch+1):
            hist = seg_model.fit_generator(self.load_training_patches(x_tr,y_tr,names_tr,z), steps_per_epoch=steps_tr, epochs=1)
            seg_model.save_weights(f'results/{self.name}/2d/seg/weights/weights-seg-{self.name}-{epoch}.h5')
            np.save(f'results/{self.name}/2d/seg/hist/{self.name}_{epoch}_seg_loss.npy', hist.history['loss'])

 
    def test_model(self, path, epoch_list=None, write_path=None, make_rec=True):

        if not epoch_list: epoch_list = [self.final_epoch]
        if write_path is None: write_path = f'results/{self.name}/2d/images/prob'
        for epoch in epoch_list:
            print(f'Testing on epoch {epoch} ...')
            seg_model = self.model_architecture()[1]
            seg_model.load_weights(f'results/{self.name}/2d/seg/weights/weights-seg-{self.name}-{epoch}.h5')
            for name_parser, x in TifReader(path):
                tran = self.transformer(x)
                x_transformed = tran.transform()
                prob_patched = seg_model.predict_generator(self.load_test_patches(x_transformed), steps=self.patches_per_img*x.shape[0]).squeeze()
                prob = convert_patches_into_image(prob_patched)
                name_parser.category = f'prob-{self.name}-{epoch}'
                TifWriter(f'{write_path}/{name_parser.split}', name_parser, prob).write()
 

def main():  
    ae = AE(name='ae', resume_epoch=200, final_epoch=200, resume_epoch_rec=20, final_epoch_rec=20, transformer=ModifiedStandardization)
    # ae.train_rec_model('data/train/images/duct', 'data/dev/images/duct')
    # ae.test_rec_model('E:/dataset/train/images/duct')
    # ae.test_rec_model('E:/dataset/dev/images/duct')
    # ae.test_rec_model('E:/dataset/val/images/duct')
    # ae.test_rec_model('E:/dataset/test/images/duct')
    # ae.train_model('data/train/images/duct', 'data/train/images/label')
    ae.test_model('E:/dataset/train/images/duct')
    ae.test_model('E:/dataset/dev/images/duct')
    ae.test_model('E:/dataset/val/images/duct')
    ae.test_model('E:/dataset/test/images/duct')

main()