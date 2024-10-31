from sklearn.model_selection import train_test_split
import tensorflow as tf
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss, total_loss
from tensorflow.keras.utils import get_custom_objects
import os
from callbacks.callbacks import get_callbacks, cosine_annealing_with_warmup
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
# from supervision.dataloader import build_augmenter, build_dataset, build_decoder
from model import build_model
import os
import tensorflow_addons as tfa
from optimizers.lion_opt import Lion

os.environ["CUDA_VISIBLE_DEVICES"]="2"

img_size = 256
BATCH_SIZE = 8
SEED = 42
save_path = "best_model.h5"

valid_size = 0.1
test_size = 0.1
epochs = 134
save_weights_only = True
max_lr = 1e-4
min_lr = 1e-6


model = build_model(img_size)
def myprint(s):
    with open('modelsummary.txt','a') as f:
        print(s, file=f)

model.summary(print_fn=myprint)
model.summary()
# model = create_segment_model()
starter_learning_rate = 1e-3
end_learning_rate = 1e-5
decay_steps = 1000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.2)

opts = tfa.optimizers.AdamW(learning_rate = 1e-4, weight_decay = learning_rate_fn)

get_custom_objects().update({"dice": dice_loss})
model.compile(optimizer = opts,
            loss='dice',
            metrics=[dice_coeff,bce_dice_loss, IoU, zero_IoU])

# model.summary()
route = 'path to training folder of created dataset'
X_path = 'path to image folder of created dataset'
Y_path = 'path to mask folder of created dataset'



X_full = sorted(os.listdir(f'{route}/image'))
Y_full = sorted(os.listdir(f'{route}/mask'))

print(len(X_full))


X_train, X_valid = train_test_split(X_full, test_size=valid_size, random_state=SEED)
Y_train, Y_valid = train_test_split(Y_full, test_size=valid_size, random_state=SEED)

#X_train, X_test = train_test_split(X_train, test_size=test_size, random_state=SEED)
#Y_train, Y_test = train_test_split(Y_train, test_size=test_size, random_state=SEED)

X_train = [X_path + x for x in X_train]
X_valid = [X_path + x for x in X_valid]
#X_test = [X_path + x for x in X_test]

Y_train = [Y_path + x for x in Y_train]
Y_valid = [Y_path + x for x in Y_valid]
#Y_test = [Y_path + x for x in Y_test]

print("N Train:", len(X_train))
print("N Valid:", len(X_valid))
#print("N test:", len(X_test))
# print(X_train)
train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder,
                            augmentAdv=False, augment=False, augmentAdvSeg=True)

valid_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
valid_dataset = build_dataset(X_valid, Y_valid, bsize=BATCH_SIZE, decode_fn=valid_decoder,
                            augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            augmentAdvSeg=False)

#test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
#test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder,
                            #augmentAdv=False, augment=False, repeat=False, shuffle=False,
                            #augmentAdvSeg=False)

callbacks = get_callbacks(monitor = 'val_loss', mode = 'min', save_path = save_path, _max_lr = max_lr
                        , _min_lr = min_lr , _cos_anne_ep = 1000, save_weights_only = save_weights_only)

steps_per_epoch = len(X_train) // BATCH_SIZE
#steps_per_epoch = 20

print("START TRAINING:")

print(train_dataset)
his = model.fit(train_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset)

model.load_weights(save_path)

#model.evaluate(test_dataset)
model.save("final_model.h5")
