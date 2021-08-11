import argparse

import torch
import numpy as np
from losses import multiview_iou_loss
from utils import AverageMeter, img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import datasets
import models
import imageio
import time
import os
from datetime import datetime
import logging

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LR_TYPE = 'step'
NUM_ITERATIONS = 250000

LAMBDA_LAPLACIAN = 5e-3
LAMBDA_FLATTEN = 5e-4

PRINT_FREQ = 100
DEMO_FREQ = 1000
SAVE_FREQ = 10000
RANDOM_SEED = 0

MODEL_DIRECTORY = 'data/results/models'
DATASET_DIRECTORY = '/mnt/e/Data/mesh_reconstruction'

IMAGE_SIZE = 64
SIGMA_VAL = 1e-4
START_ITERATION = 0

RESUME_PATH = ''

date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
os.makedirs("logs", exist_ok=True)
log_filename = "logs/continual-learning-{}.log".format(date_time)

# create the log file if it does not exist
if not os.path.isfile(log_filename):
    open(log_filename, "w").close()

logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(dataset_train, model, optimizer, directory_output, image_output, args):
    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # loss function
    loss_fn = nn.CrossEntropy()

    for i in range(START_ITERATION, args.num_iterations + 1):
        # adjust learning rate and sigma_val (decay after 150k iter)
        lr = adjust_learning_rate(
            [optimizer], args.learning_rate, i, method=args.lr_type)
        model.set_sigma(adjust_sigma(args.sigma_val, i))

        # load images from multi-view
        images_a, images_b, viewpoints_a, viewpoints_b, labels = dataset_train.get_random_batch(args.batch_size)
        images_a = images_a.cuda()
        images_b = images_b.cuda()
        viewpoints_a = viewpoints_a.cuda()
        viewpoints_b = viewpoints_b.cuda()

        # soft render images
        render_images, laplacian_loss, flatten_loss, class_predictions = model([images_a, images_b],
                                                                               [viewpoints_a,
                                                                                viewpoints_b],
                                                                               task='train')

        # compute classification lsos
        ce_loss = loss_fn(class_predictions, labels)
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        ewc_loss = model.ewc_loss(cuda=True)

        # compute loss
        loss = multiview_iou_loss(render_images, images_a, images_b) + \
            args.lambda_laplacian * laplacian_loss + \
            args.lambda_flatten * flatten_loss + ewc_loss + ce_loss
        losses.update(loss.data.item(), images_a.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # save checkpoint
        if i % args.save_freq == 0:
            model_path = os.path.join(
                directory_output, 'checkpoint_%07d.pth.tar' % i)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_path)

        # save demo images
        if i % args.demo_freq == 0:
            demo_image = images_a[0:1]
            demo_path = os.path.join(directory_output, 'demo_%07d.obj' % i)
            demo_v, demo_f = model.reconstruct(demo_image)
            srf.save_obj(demo_path, demo_v[0], demo_f[0])

            imageio.imsave(os.path.join(image_output, '%07d_fake.png' %
                           i), img_cvt(render_images[0][0]))
            imageio.imsave(os.path.join(
                image_output, '%07d_input.png' % i), img_cvt(images_a[0]))

        # print
        if i % args.print_freq == 0:
            logger.info('Iter: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f}\t'
                        'Loss {loss.val:.3f}\t'
                        'lr {lr:.6f}\t'
                        'sv {sv:.6f}\t'.format(i, args.num_iterations,
                                               batch_time=batch_time, loss=losses,
                                               lr=lr, sv=model.rasterizer.sigma_val))


def test(dataset_val, model, directory_mesh):
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()

    iou_all = []

    for class_id, class_name in dataset_val.class_ids_pair:

        directory_mesh_cls = os.path.join(directory_mesh, class_id)
        os.makedirs(directory_mesh_cls, exist_ok=True)
        iou = 0

        for i, (im, vx) in enumerate(dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)):
            images = torch.autograd.Variable(im).cuda()
            voxels = vx.numpy()

            batch_iou, vertices, faces = model(
                images, voxels=voxels, task='test')
            iou += batch_iou.sum()

            batch_time.update(time.time() - end)
            end = time.time()

            # save demo images
            for k in range(vertices.size(0)):
                obj_id = (i * args.batch_size + k)
                if obj_id % args.save_freq == 0:
                    mesh_path = os.path.join(
                        directory_mesh_cls, '%06d.obj' % obj_id)
                    input_path = os.path.join(
                        directory_mesh_cls, '%06d.png' % obj_id)
                    srf.save_obj(mesh_path, vertices[k], faces[k])
                    imageio.imsave(input_path, img_cvt(images[k]))

            # print loss
            if i % args.print_freq == 0:
                logger.info('Iter: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f}\t'
                            'IoU {2:.3f}\t'.format(i, ((dataset_val.num_data[class_id] * 24) // args.batch_size),
                                                   batch_iou.mean(),
                                                   batch_time=batch_time))

        iou_cls = iou / 24. / dataset_val.num_data[class_id] * 100
        iou_all.append(iou_cls)
        logger.info('=================================')
        logger.info('Mean IoU: %.3f for class %s' % (iou_cls, class_name))
        logger.info('\n')

    logger.info('=================================')
    logger.info('Mean IoU: %.3f for all classes' % (sum(iou_all) / len(iou_all)))


def adjust_learning_rate(optimizers, learning_rate, i, method):
    if method == 'step':
        lr, decay = learning_rate, 0.3
        if i >= 150000:
            lr *= decay
    elif method == 'constant':
        lr = learning_rate
    else:
        logger.info("no such learing rate type")

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_sigma(sigma, i):
    decay = 0.3
    if i >= 150000:
        sigma *= decay
    return sigma


if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--experiment-id', type=str)
    parser.add_argument('-md', '--model-directory',
                        type=str, default=MODEL_DIRECTORY)
    parser.add_argument('-r', '--resume-path', type=str, default=RESUME_PATH)
    parser.add_argument('-dd', '--dataset-directory',
                        type=str, default=DATASET_DIRECTORY)
    parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
    parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
    parser.add_argument('-b', '--batch-size', type=int, default=BATCH_SIZE)

    parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

    parser.add_argument('-lr', '--learning-rate',
                        type=float, default=LEARNING_RATE)
    parser.add_argument('-lrt', '--lr-type', type=str, default=LR_TYPE)

    parser.add_argument('-ll', '--lambda-laplacian',
                        type=float, default=LAMBDA_LAPLACIAN)
    parser.add_argument('-lf', '--lambda-flatten',
                        type=float, default=LAMBDA_FLATTEN)
    parser.add_argument('-ni', '--num-iterations',
                        type=int, default=NUM_ITERATIONS)
    parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
    parser.add_argument('-df', '--demo-freq', type=int, default=DEMO_FREQ)
    parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--fisher_estimation_sample_size', type=int, default=1024)

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # setup model & optimizer
    model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.model_param(), args.learning_rate)

    # train output directories
    directory_output = os.path.join(args.model_directory, args.experiment_id, date_time)
    os.makedirs(directory_output, exist_ok=True)
    image_output = os.path.join(directory_output, 'pic')
    os.makedirs(image_output, exist_ok=True)

    # test output directories
    directory_mesh = os.path.join(directory_output, 'test')
    os.makedirs(directory_mesh, exist_ok=True)
    num_set = 4
    class_ids = args.class_ids.split(',')

    # exclude one class to make 4 sets of 3 classes
    class_ids.pop()

    # TODO: set an argument for the number of classes in
    train_ids = val_ids = [class_ids.pop(), class_ids.pop(), class_ids.pop()]

    while num_set:
        # display current classes that we are training/validating on
        logger.info(
            f"Training on {train_ids} which correspond to {[datasets.class_ids_map.get(train_id) for train_id in train_ids]}")
        logger.info(
            f"Validating on {val_ids} which correspond to {[datasets.class_ids_map.get(val_id) for val_id in val_ids]}")

        model.train()
        dataset_train = datasets.ShapeNet(
            args.dataset_directory, train_ids, 'train')
        train(dataset_train=dataset_train, model=model,
              optimizer=optimizer, directory_output=directory_output, image_output=image_output, args=args)
        model.consolidate(model.estimate_fisher(
            dataset_train, args.fisher_estimation_sample_size
        ))

        # delete train dataset to free memory
        del dataset_train

        model.eval()
        dataset_val = datasets.ShapeNet(args.dataset_directory, val_ids, 'val')
        test(dataset_val, model, directory_mesh=directory_mesh)

        num_set -= 1
        if num_set:
            new_class1 = class_ids.pop()
            new_class2 = class_ids.pop()
            new_class3 = class_ids.pop()
            train_ids = [new_class1, new_class2, new_class3]
            val_ids.append(new_class1)
            val_ids.append(new_class2)
            val_ids.append(new_class3)
