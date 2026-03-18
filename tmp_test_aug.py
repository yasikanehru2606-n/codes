import os, cv2, numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

src='captured_data\\A\\a_small_0.png'
print('src', src, os.path.isfile(src))
img=cv2.imread(src)
print('img', img is not None, img.shape if img is not None else None)
if img is None:
    raise SystemExit('img load failed')
img=cv2.resize(img,(128,128))
img=np.expand_dims(img,axis=0)
out='tmp_aug_test'
os.makedirs(out, exist_ok=True)

if os.path.isdir(out):
    for f in os.listdir(out):
        os.remove(os.path.join(out,f))

datagen=ImageDataGenerator(rotation_range=25,width_shift_range=0.2,height_shift_range=0.2,zoom_range=0.2,shear_range=0.15,horizontal_flip=True,brightness_range=[0.7,1.3],fill_mode='nearest')

i=0
for batch in datagen.flow(img,batch_size=1,save_to_dir=out,save_prefix='aug',save_format='png'):
    i+=1
    if i>=5:
        break

print('generated', len([f for f in os.listdir(out) if f.startswith('aug')]))
print('files', os.listdir(out))
