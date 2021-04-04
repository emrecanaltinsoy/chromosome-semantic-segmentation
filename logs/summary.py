import tensorflow as tf
import os
import yaml
import glob

files = glob.glob("./scalar/**/event*", recursive = True)

for f in files:
    file_path = os.path.join(f.split('\\')[0],f.split('\\')[1])
    file_name = os.path.join(f.split('\\')[-2],f.split('\\')[-1])

    i=0
    loss = {}
    val_loss = {}

    for e in tf.compat.v1.train.summary_iterator(os.path.join(file_path,file_name)):
        for v in e.summary.value:
            if v.tag == 'orig_loss':
                i += 1
                if i > 40: break
                loss[i] = v.simple_value
            if v.tag == 'orig_val_loss':
                if i > 40: break
                val_loss[i] = v.simple_value

    losses = {
        'loss': loss,
        'val_loss': val_loss,
    }
    with open(os.path.join(file_path,'losses.yaml'), "w") as fp:
        yaml.dump(losses, fp)
