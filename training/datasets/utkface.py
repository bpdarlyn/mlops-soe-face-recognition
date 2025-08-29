import os, csv, tensorflow as tf

def load_csv(csv_path):
    items = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            items.append((row["filename"], int(row["age"]), int(row["gender"])))
    return items

def make_datasets(root_dir, img_size=(160,160), batch=32, val_split=0.1, shuffle=1000):
    data_dir = os.path.join(root_dir, "data", "UTKFace")
    csv_path = os.path.join(data_dir, "labels.csv")
    items = load_csv(csv_path)
    n = len(items)
    n_val = int(n * val_split)
    train_items = items[n_val:]
    val_items   = items[:n_val]

    def _parse(fname, age, gender):
        img_path = tf.strings.join([data_dir, "/", fname])
        img = tf.io.read_file(img_path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        # targets: edad (regresión) y género (binario)
        return img, {"age": tf.cast(age, tf.float32), "gender": tf.cast(gender, tf.float32)}

    def _to_ds(samples):
        # Separar los datos en componentes homogéneos
        filenames = [item[0] for item in samples]
        ages = [item[1] for item in samples]
        genders = [item[2] for item in samples]
        
        ds = tf.data.Dataset.from_tensor_slices((filenames, ages, genders))
        ds = ds.shuffle(shuffle)
        ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
        return ds

    return _to_ds(train_items), _to_ds(val_items)
