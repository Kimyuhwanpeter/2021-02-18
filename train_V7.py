# -*- coding:utf-8 -*-
from random import random, shuffle
from absl import flags
from model_V7 import *

import matplotlib.pyplot as plt
import sys
import os

#gpus = tf.config.experimental.list_physical_devices('gpu')
#if gpus:
#  # 텐서플로가 첫 번째 gpu에 1gb 메모리만 할당하도록 제한
#  try:
#    tf.config.experimental.set_virtual_device_configuration(
#        gpus[0],
#        [tf.config.experimental.virtualdeviceconfiguration(memory_limit=7024)])
#  except runtimeerror as e:
#    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
#    print(e)

flags.DEFINE_string("in_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/male_16_39_train.txt", "Input text path")

flags.DEFINE_string("se_txt_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-M_Morph-F_16_39_40_63/train/female_40_63_train.txt", "Style text path")

flags.DEFINE_string("in_img_path", "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/", "Input image path")

flags.DEFINE_string("se_img_path", "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/female_40_63/", "Style image path")

flags.DEFINE_integer("n_classes", 24, "Number of classes")

flags.DEFINE_integer("img_size", 256, "Input image size")

flags.DEFINE_integer("load_size", 266, "Original image size")

flags.DEFINE_integer("batch_size", 1, "Training batch size")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_float("lr", 0.0002, "Learning rate")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path (test)")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path (train)")

flags.DEFINE_string("save_images", "", "Save sample images")


FLAGS = flags.FLAGS
FLAGS(sys.argv)

ge_optim = tf.keras.optimizers.Adam(FLAGS.lr)
se_optim = tf.keras.optimizers.Adam(FLAGS.lr)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr)

def input_img_map(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.

    if random() < 0.5:
        img = tf.image.flip_left_right(img)

    lab = lab_list - 16 + 1
    nor = lab / FLAGS.n_classes

    return img, lab, nor

def style_img_map(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.

    if random() < 0.5:
        img = tf.image.flip_left_right(img)

    lab = lab_list - 40 + 1
    nor = lab / FLAGS.n_classes

    style_img = tf.keras.layers.GaussianNoise(float(1-nor + 1e-5))(img, True) # A -> B
    style_img2 = tf.keras.layers.GaussianNoise(float(nor + 1e-5))(img, True) # A -> B -> A

    return img, lab, nor, style_img, style_img2


def cal_loss(input_images, style_images, noise_img, noise_img2, ge_model, se_model, dis_model):

    with tf.GradientTape(persistent=True) as tape:
        style_1, style_2, style_3 = se_model(style_images, True)    # include gender and race
        n_style_1, n_style_2, n_style_3 = se_model(noise_img, True)    # age
        n1_style_1, n2_style_2, n3_style_3 = se_model(noise_img2, True)    # age

        fake_img = ge_model([input_images, style_1, style_2, style_3], True)
        cycle_fake_img = ge_model([fake_img, n1_style_1, n2_style_2, n3_style_3], True)
        fake_n_img = ge_model([input_images, n_style_1, n_style_2, n_style_3], True)

        real_dis = dis_model(input_images, True)
        fake_dis = dis_model(fake_img, True)

        g_style_reconstruct = tf.reduce_mean(tf.abs(fake_img - style_images))
        g_style_diver = tf.reduce_mean(tf.abs(fake_n_img - fake_img))
        g_cycle = tf.reduce_mean(tf.abs(input_images - cycle_fake_img))
        g_ad = tf.reduce_mean((fake_dis - tf.ones_like(fake_dis))**2)
        g_loss = g_style_reconstruct + g_style_diver + g_cycle + g_ad
        
        d_loss = (tf.reduce_mean((real_dis - tf.ones_like(real_dis))**2) \
            + tf.reduce_mean((fake_dis - tf.zeros_like(fake_dis))**2)) / 2

    ge_grads = tape.gradient(g_loss, ge_model.trainable_variables)
    se_grads = tape.gradient(g_loss, se_model.trainable_variables)
    de_grads = tape.gradient(d_loss, dis_model.trainable_variables)

    ge_optim.apply_gradients(zip(ge_grads, ge_model.trainable_variables))
    se_optim.apply_gradients(zip(se_grads, se_model.trainable_variables))
    d_optim.apply_gradients(zip(de_grads, dis_model.trainable_variables))

    return g_loss, d_loss

def main():
    # 한 이미지에 대해 입력하고, 나이대 별로 이미지를 비교하여 학습?
    # 나이대 별로 이미지를 넣고 loss를 적용할 때 normalization 된 나이 값을 같이 준다.
    # 그렇게하면 테스트할 때는 나이 값을 주면 입력 이미지가 그 나이 값에 대해서 변환
    # 생각은 단순하고 좋은것같다. 그러러면 필요한것은 generator 모델과  style관련 모델과 discirminator 모델이 필요하다.
    # 16세 이미지를 넣었으면 --> 17세부터 맨 끝 나이까지 (랜덤으로 선택)
    # 기억하자!!!!

    ge_model = V7_generator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                            style_shape_1=(FLAGS.img_size, FLAGS.img_size, 64),
                            style_shape_2=(FLAGS.img_size // 2, FLAGS.img_size // 2, 128),
                            style_shape_3=(FLAGS.img_size // 4, FLAGS.img_size // 4, 256))
    se_model = style_map(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    dis_model = discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    ge_model.summary()
    se_model.summary()
    dis_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(ge_model=ge_model,
                                   se_model=se_model,
                                   dis_model=dis_model,
                                   ge_optim=ge_optim,
                                   se_optim=se_optim,
                                   d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!!")

    if FLAGS.train:
        count = 0

        input_img = np.loadtxt(FLAGS.in_txt_path, dtype="<U100", skiprows=0, usecols=0)
        input_img = [FLAGS.in_img_path + img for img in input_img]
        input_lab = np.loadtxt(FLAGS.in_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        style_img = np.loadtxt(FLAGS.se_txt_path, dtype="<U100", skiprows=0, usecols=0)
        style_img = [FLAGS.se_img_path + img for img in style_img]
        style_lab = np.loadtxt(FLAGS.se_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        for epoch in range(FLAGS.epochs):

            T = list(zip(input_img, input_lab))
            shuffle(T)
            input_img, input_lab = zip(*T)
            input_img, input_lab = np.array(input_img), np.array(input_lab)

            S = list(zip(style_img, style_lab))
            shuffle(S)
            style_img, style_lab = zip(*T)
            style_img, style_lab = np.array(style_img), np.array(style_lab)

            mapping_input = tf.data.Dataset.from_tensor_slices((input_img, input_lab))
            mapping_input = mapping_input.shuffle(len(input_img))
            mapping_input = mapping_input.map(input_img_map)
            mapping_input = mapping_input.batch(FLAGS.batch_size)
            mapping_input = mapping_input.prefetch(tf.data.experimental.AUTOTUNE)

            mapping_style = tf.data.Dataset.from_tensor_slices((style_img, style_lab))
            mapping_style = mapping_style.shuffle(len(style_img))
            mapping_style = mapping_style.map(style_img_map)
            mapping_style = mapping_style.batch(FLAGS.batch_size)
            mapping_style = mapping_style.prefetch(tf.data.experimental.AUTOTUNE)

            input_iter = iter(mapping_input)
            style_iter = iter(mapping_style)
            train_idx = min(len(input_img), len(style_img)) // FLAGS.batch_size
            for step in range(train_idx):

                input_images, input_labels, labels_nom = next(input_iter)
                style_images, style_labels, style_nom, noise_img, noise_img2 = next(style_iter)

                g_loss, d_loss = cal_loss(input_images, style_images, noise_img, noise_img2, ge_model, se_model, dis_model)

                print(g_loss, d_loss)


if __name__ == "__main__":
    main()
