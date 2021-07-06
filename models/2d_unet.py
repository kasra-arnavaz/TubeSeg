from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Conv2DTranspose, Cropping2D
from tensorflow.keras.optimizers import Adam, SGD
from utils.data_utils import *
from utils.patch_utils import *
from utils.transform_data import *
from cldice_loss.cldice import soft_dice_cldice_loss

class UNet:

    input_size = 320
    output_size = 256
    image_size = 1024
    margin = int((input_size -output_size)/2)
    grid_size = int(image_size/output_size)
    patches_per_img = grid_size**2

    def __init__(self, name, resume_epoch, final_epoch, transformer: TransformData, batch_size=8):
        self.name = name
        self.resume_epoch = resume_epoch
        self.final_epoch = final_epoch
        self.batch_size = batch_size
        self.transformer = transformer

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
            y_gen = np.zeros([self.batch_size, self.output_size, self.output_size])
            mask = np.random.randint(num_samples, size=self.batch_size)
            for i,m in enumerate(mask):
                x_gen[i], y_gen[i] = make_valid_patch(x[m], names[m], y[m])
                deg = np.random.randint(4)
                x_gen[i] = np.rot90(x_gen[i], deg, (0,1))
                y_gen[i] = np.rot90(y_gen[i], deg, (0,1))
            x_gen = np.expand_dims(x_gen, axis=-1)
            y_gen = np.expand_dims(y_gen, axis=-1)
            yield x_gen, y_gen


    def train_model(self, x_path, y_path):
        my_model = self.model_architecture()
        my_model.compile(optimizer=Adam(learning_rate=1e-4), loss=soft_dice_cldice_loss, metrics = ['binary_accuracy'])
        x_tr, y_tr, names_tr, z_list = self.extract_training_images(x_path, y_path)
        num_samples = x_tr.shape[0] * self.patches_per_img
        steps_tr = np.ceil(num_samples / self.batch_size)
        if self.resume_epoch>0:
            print(f'Resuming training on epoch {self.resume_epoch} ...')
            my_model.load_weights(f'results/{self.name}/2d/weights/{self.name}_{self.resume_epoch}_weights.h5')
        else: print('Training from scratch ...')
        for epoch in np.arange(self.resume_epoch+1, self.final_epoch+1):
            hist = my_model.fit_generator(self.load_training_patches(x_tr,y_tr,names_tr,z_list), steps_per_epoch=steps_tr, epochs=1)
            os.makedirs(f'results/{self.name}/2d/weights', exist_ok=True)
            my_model.save_weights(f'results/{self.name}/2d/weights/weights-{self.name}-{epoch}.h5')
            os.makedirs(f'results/{self.name}/2d/hist', exist_ok=True)
            np.save(f'results/{self.name}/2d/hist/{self.name}_{epoch}_seg_loss.npy', hist.history['loss'])

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
        if write_path is None: write_path = f'results/{self.name}/2d/images/prob'
        for epoch in epoch_list:
            print(f'Testing on epoch {epoch} ...')
            my_model = self.model_architecture()
            my_model.load_weights(f'results/{self.name}/2d/weights/weights-{self.name}-{epoch}.h5')
            for name_parser, x in TifReader(path):
                x_transformed = self.transformer(x).transform()
                prob_patched = my_model.predict_generator(self.load_test_patches(x_transformed), steps=self.patches_per_img*x.shape[0]).squeeze()
                prob = convert_patches_into_image(prob_patched)
                name_parser.category = f'prob-{self.name}-{epoch}'
                TifWriter(f'{write_path}/{name_parser.split}', name_parser, prob).write()

def main():
    unet = UNet(name='unetcldice', resume_epoch=200, final_epoch=200, transformer=ModifiedStandardization)
    unet.train_model('data/train/images/ducts', 'data/train/images/labels')
    unet.test_model('E:/dataset/train/images/duct')
    unet.test_model('E:/dataset/dev/images/duct')
    unet.test_model('E:/dataset/val/images/duct')
    unet.test_model('E:/dataset/test/images/duct')

main()