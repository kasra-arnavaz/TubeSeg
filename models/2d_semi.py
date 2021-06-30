from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from utils.data_utils import *
from utils.patch_utils import *
from utils.transform_data import *

class SemiSupervised:

    input_size = 320
    output_size = 256
    image_size = 1024
    margin = int((input_size -output_size)/2)
    grid_size = int(image_size/output_size)
    patches_per_img = grid_size**2

    def __init__(self, name, resume_epoch, final_epoch, loss_weights, transformer: TransformData, batch_size=8):
        self.name = name
        self.resume_epoch = resume_epoch
        self.final_epoch = final_epoch
        self.batch_size = batch_size
        self.transformer = transformer
        self.loss_weights = loss_weights


    def model_architecture(self):
        
        input1 = Input((self.input_size,self.input_size,1))
        conv1 = Conv2D(8, 3, activation='relu', padding='same')(input1)
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
        out_rec = Cropping2D((self.margin,self.margin), name='rec')(conv17)
 
########################################
        
        conv18 = Conv2D(64, 3, activation='relu', padding='same')(maxpool3)
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
        out2 = Cropping2D((self.margin,self.margin), name='seg')(conv26)

        model = Model(input1, [out_rec, out2])
        # print(seg_model.summary())

        return model

    def extract_training_images(self, x_path, y_path, dev_path):
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
        y_tr = np.clip(y_tr, 0, 1) # because the intensity for foreground images are sometimes 128 or 255.
        for name_parser, x in TifReader(dev_path):
            x_transformed = self.transformer(x).transform()
            y = -1*np.ones_like(x)
            x_tr = np.append(x_tr, x_transformed, axis=0)
            y_tr = np.append(y_tr, y, axis=0)
            names_tr.append(name_parser.name)
            z_list.append(x.shape[0])
        x_tr, y_tr = x_tr[1:], y_tr[1:]
        return x_tr, y_tr, names_tr, z_list


    def load_training_patches(self, x, y, names_tr, z_list, frac=0.25):
        
        while True:
            names = []
            for i, name in enumerate(names_tr):
                names += [name]*z_list[i]
            num_samples = x.shape[0]
            x_gen = np.zeros([self.batch_size, self.input_size, self.input_size])
            target_gen = np.zeros([2, self.batch_size, self.output_size, self.output_size])
            mask_L = np.random.randint(int(num_samples*frac), size=int(np.ceil(self.batch_size*frac)))
            mask_U = np.random.randint(int(num_samples*frac), num_samples, int(np.floor(self.batch_size*(1-frac))))
            mask = np.append(mask_L, mask_U)
            np.random.shuffle(mask)
            for i, m in enumerate(mask):
                x_gen[i], target_gen[:,i] = make_valid_patch(x[m], names[m], y[m])
                deg = np.random.randint(4)
                x_gen[i] = np.rot90(x_gen[i], deg, (0,1))
                target_gen[:,i] = np.rot90(target_gen[:,i], deg, (1,2))
            x_gen = np.expand_dims(x_gen, axis=-1)
            target_gen = np.expand_dims(target_gen, axis=-1)
            is_labeled = np.zeros(self.batch_size)
            for i in range(self.batch_size):
               is_labeled[i] = np.any(target_gen[1,i] != -1)
            yield (x_gen, [target_gen[0], target_gen[1]], [np.ones(self.batch_size),is_labeled*1])

 
    def train_model(self, x_path, y_path, dev_path):

        os.makedirs(f'results/{self.name}/2d/weights', exist_ok=True)
        os.makedirs(f'results/{self.name}/2d/hist', exist_ok=True)
        my_model = self.model_architecture()
        my_model.compile(optimizer=Adam(learning_rate=1e-4), loss=['mse', 'binary_crossentropy'], loss_weights=self.loss_weights, metrics={'seg':'binary_accuracy'})
        x_tr, y_tr, names_tr, z = self.extract_training_images(x_path, y_path, dev_path)
        num_samples = x_tr.shape[0] * self.patches_per_img
        steps_tr = np.ceil(num_samples / self.batch_size)
        if self.resume_epoch>0:
            print(f'Resuming training on epoch {self.resume_epoch} ...')
            my_model.load_weights(f'results/{self.name}/2d/weights/weights-{self.name}-{self.resume_epoch}.h5')
        else: print('Training from scratch ...')
        for epoch in np.arange(self.resume_epoch+1, self.final_epoch+1):
            hist = my_model.fit_generator(self.load_training_patches(x_tr,y_tr,names_tr,z), steps_per_epoch=steps_tr, epochs=1)
            my_model.save_weights(f'results/{self.name}/2d/weights/weights-{self.name}-{epoch}.h5')
            np.save(f'results/{self.name}/2d/hist/{self.name}_{epoch}_combined_loss.npy', hist.history['loss'])
            np.save(f'results/{self.name}/2d/hist/{self.name}_{epoch}_seg_loss.npy', hist.history['seg_loss'])
            np.save(f'results/{self.name}/2d/hist/{self.name}_{epoch}_rec_loss.npy', hist.history['rec_loss'])


    def load_test_patches(self, x):
        x_pad = np.pad(x, ((0,0),(self.margin,self.margin),(self.margin,self.margin)), 'constant', constant_values=0)
        x_pad = np.expand_dims(x_pad, axis=-1)
        for z in range(x.shape[0]):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    yield x_pad[np.newaxis, z, i*self.output_size:i*self.output_size+self.input_size,\
                                                j*self.output_size:j*self.output_size+self.input_size,:]


    def test_model(self, path, epoch_list=None, write_path=None, make_rec=True):

        if epoch_list is None: epoch_list = [self.final_epoch]
        if write_path is None: write_path = f'results/{self.name}/2d/images'
        for epoch in epoch_list:
            print(f'Testing on epoch {epoch} ...')
            my_model = self.model_architecture()
            my_model.load_weights(f'results/{self.name}/2d/weights/weights-{self.name}-{epoch}.h5')
            for name_parser, x in TifReader(path):
                tran = self.transformer(x)
                x_transformed = tran.transform()
                rec_patched, prob_patched = my_model.predict_generator(self.load_test_patches(x_transformed), steps=self.patches_per_img*x.shape[0])
                prob = convert_patches_into_image(prob_patched.squeeze())
                name_parser.category = f'prob-{self.name}-{epoch}'
                TifWriter(f'{write_path}/prob', name_parser, prob).write()
                if make_rec:
                    rec = convert_patches_into_image(rec_patched.squeeze())
                    rec = (rec*3*tran.std) + tran.mean
                    rec = rec.astype('uint16')
                    name_parser.category = f'rec-{self.name}-{epoch}'
                    TifWriter(f'{write_path}/rec', name_parser, rec).write()

def main():
    import glob
    semi = SemiSupervised(name='semi', resume_epoch=40, final_epoch=40, loss_weights=[1, 10], transformer=ModifiedStandardization)
    for path in glob.glob('movie/val/*'):
        os.remove(f'{path}/prob')
        os.remove(f'{path}/pred')
        semi.test_model(f'{path}/mcherry', write_path = path)

main()
