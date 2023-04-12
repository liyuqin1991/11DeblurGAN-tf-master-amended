IMPORT
PIL.Image
IMPORT
numpy as np
IMPORT
random
IMPORT
math
IMPORT
os

FUNCTION
image_loader(image_path, load_x, load_y, is_train=True)
imgs = sorted(os.listdir(image_path))
img_list = []
FOR
each
ele in imgs
DO
img = Image.open(os.path.join(image_path, ele))
IF
is_train
THEN
img = img.resize((load_x, load_y), Image.BICUBIC)
END
IF
img_list.append(np.array(img))
END
FOR
RETURN
img_list
END
FUNCTION

FUNCTION
data_augument(lr_img, hr_img, aug)
IF
aug < 4
THEN
lr_img = np.rot90(lr_img, aug)
hr_img = np.rot90(hr_img, aug)
ELSE
IF
aug == 4
THEN
lr_img = np.fliplr(lr_img)
hr_img = np.fliplr(hr_img)
ELSE
IF
aug == 5
THEN
lr_img = np.flipud(lr_img)
hr_img = np.flipud(hr_img)
ELSE
IF
aug == 6
THEN
lr_img = np.rot90(np.fliplr(lr_img))
hr_img = np.rot90(np.fliplr(hr_img))
ELSE
IF
aug == 7
THEN
lr_img = np.rot90(np.flipud(lr_img))
hr_img = np.rot90(np.flipud(hr_img))
END
IF
RETURN
lr_img, hr_img
FUNCTION
batch_gen(blur_imgs, sharp_imgs, patch_size, batch_size, random_index, step, augment)
img_index = random_index[step * batch_size: (step + 1) * batch_size]
all_img_blur = []
all_img_sharp = []
FOR
each
_index in img_index
DO
all_img_blur.append(blur_imgs[_index])
all_img_sharp.append(sharp_imgs[_index])
END
FOR

scss
Copy
code
blur_batch = []
sharp_batch = []
FOR
i
from

0
to
length(all_img_blur) - 1
DO
ih, iw, _ = all_img_blur[i].shape
ix = random.randrange(0, iw - patch_size + 1)
iy = random.randrange(0, ih - patch_size + 1)

img_blur_in = all_img_blur[i][iy:iy + patch_size, ix:ix + patch_size]
img_sharp_in = all_img_sharp[i][iy:iy + patch_size, ix:ix + patch_size]

IF
augment
THEN
aug = random.randrange(0, 8)
img_blur_in, img_sharp_in = data_augument(img_blur_in, img_sharp_in, aug)
END
IF

blur_batch.append(img_blur_in)
sharp_batch.append(img_sharp_in)
END
FOR

blur_batch = np.array(blur_batch)
sharp_batch = np.array(sharp_batch)

RETURN
blur_batch, sharp_batch

FUNCTION
recursive_forwarding(blur, chopSize, session, net_model, chopShave=20):
b, h, w, c = SHAPE(blur)
wHalf = FLOOR(w / 2)
hHalf = FLOOR(h / 2)

wc = wHalf + chopShave
hc = hHalf + chopShave

inputPatch = ARRAY(
    (blur[:, :hc, :wc, :], blur[:, :hc, (w - wc):, :], blur[:, (h - hc):, :wc, :], blur[:, (h - hc):, (w - wc):, :]))
outputPatch = []
IF
wc * hc < chopSize
THEN
FOR
ele
IN
inputPatch
DO
output = session.run(net_model.output, feed_dict={net_model.blur: ele})
APPEND
outputPatch
WITH
output
ENDFOR
ELSE
FOR
ele
IN
inputPatch
DO
output = recursive_forwarding(ele, chopSize, session, net_model, chopShave)
APPEND
outputPatch
WITH
output
ENDFOR
ENDIF

upper = CONCATENATE((outputPatch[0][:, :hHalf, :wHalf, :], outputPatch[1][:, :hHalf, wc - w + wHalf:, :]), axis=2)
rower = CONCATENATE(
    (outputPatch[2][:, hc - h + hHalf:, :wHalf, :], outputPatch[3][:, hc - h + hHalf:, wc - w + wHalf:, :]), axis=2)
output = CONCATENATE((upper, rower), axis=1)

RETURN
output
ENDFUNCTION