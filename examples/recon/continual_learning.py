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
NUM_ITERATIONS_CLASSIFIER = 50000

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


def train(dataset_train, model, optimizer, directory_output, image_output, start_iter, args):
    end = time.time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for i in range(start_iter, args.num_iterations + 1):
        # adjust learning rate and sigma_val (decay after 150k iter)
        lr = adjust_learning_rate(
            [optimizer], args.learning_rate, i, method=args.lr_type)
        model.set_sigma(adjust_sigma(args.sigma_val, i))

        # load images from multi-view
        images_a, images_b, viewpoints_a, viewpoints_b, labels = [
            tensor.to(args.device) for tensor in dataset_train.get_random_batch(args.batch_size)]

        # soft render images
        render_images, laplacian_loss, flatten_loss, _ = model(images=[images_a, images_b],
                                                               viewpoints=[viewpoints_a,
                                                                           viewpoints_b],
                                                               task='train')
        # TODO inspect render images
        # compute classification lsos
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        # ewc_loss = model.ewc_loss(cuda=True)

        # compute loss
        loss = multiview_iou_loss(render_images, images_a, images_b) + \
            args.lambda_laplacian * laplacian_loss + \
            args.lambda_flatten * flatten_loss  # + ewc_loss

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
                directory_output, 'checkpoint_%07d.pkl' % i)
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

        if i % args.print_freq == 0:
            log_str = 'Iter: [{0}/{1}]\tTime {batch_time.val:.3f}\tLoss {loss.val:.3f}\tlr {lr:.6f}\tsv {sv:.6f}\t'.format(i, args.num_iterations,
                                                                                                                           batch_time=batch_time, loss=losses, lr=lr, sv=model.rasterizer.sigma_val)
            logger.info(log_str)

    # 2nd traning phase which focus on training the 3D shape classifier
    # loss function
    classification_loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(1, args.num_iterations_classifier + 1):
        with torch.no_grad():
            # load images from multi-view
            images_a, images_b, viewpoints_a, viewpoints_b, labels = [
                tensor.to(args.device) for tensor in dataset_train.get_random_batch(args.batch_size)]

            # soft render images
            _, _, _, silhouettes = model(images=[images_a, images_b],
                                         viewpoints=[viewpoints_a,
                                                     viewpoints_b],
                                         task='train')

        # compute classification loss
        rendered_images = silhouettes.detach().clone()
        rendered_images.requires_grad = True

        # only taking first three channels as the 4th channel is a mask
        class_predictions = model.mvcnn(rendered_images[:, :3, :, :])

        # compute cross entropy loss
        ce_loss = classification_loss_fn(class_predictions, labels)

        # accuracy
        _max_values, max_indices = torch.max(class_predictions, dim=1)
        batch_acc = torch.sum(max_indices == labels) / args.batch_size

        # ewc_loss = model.ewc_loss(cuda=True)

        # compute gradient and optimize
        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_str = 'Iter: [{0}/{1}]\tTime {batch_time.val:.3f}\tCE Loss {ce_loss:.3f}\tAccuracy {acc:.3f}'.format(i, args.num_iterations_classifier,
                                                                                                                     batch_time=batch_time, ce_loss=ce_loss.item(), acc=batch_acc.item())
            logger.info(log_str)


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
        acc = []
        ce_losses = []

        for i, (im, vx) in enumerate(dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)):
            images = torch.autograd.Variable(im).to(device)
            voxels = vx.numpy()

            with torch.no_grad():
                batch_iou, vertices, faces, silhouettes = model(
                    images, voxels=voxels, task='test')
                iou += batch_iou.sum()

                # only taking first three channels as the 4th channel is a mask
                class_predictions = model.mvcnn(silhouettes[:, :3, :, :])

            labels = torch.empty(class_predictions.shape[0]).fill_(dataset_val.class_ids_labels.get(class_id))

            # compute cross entropy loss
            ce_loss = classification_loss_fn(class_predictions, labels)

            # accuracy
            _max_values, max_indices = torch.max(class_predictions)
            batch_acc = torch.sum(max_indices == labels) / args.batch_size
            acc.append(batch_acc.item())
            ce_losses.append(ce_loss.item())

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
                            'Time: {batch_time.val:.3f}\t'
                            'IoU: {2:.3f}\tCE Loss: {ce_loss:.3f}\tBatch accuracy: {batch_acc:.3f}'.format(i, ((dataset_val.num_data[class_id] * 24) // args.batch_size),
                                                                                                           batch_iou.mean(),
                                                                                                           batch_time=batch_time, ce_loss=ce_loss.item(), batch_acc=batch_acc.item()))

        iou_cls = iou / 24. / dataset_val.num_data[class_id] * 100
        iou_all.append(iou_cls)
        logger.info('=================================')
        logger.info('Mean IoU: %.3f for class %s' % (iou_cls, class_name))
        logger.info('Mean Accuracy: %.3f for class %s' % (sum(batch_acc)/len(batch_acc), class_name))
        logger.info('Mean CE Loss: %.3f for class %s' % (sum(ce_losses)/len(ce_losses), class_name))
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
    parser.add_argument('-lrc', '--learning-rate-classifier',
                        type=float, default=LEARNING_RATE_CLASSIFIER)
    parser.add_argument('-lrt', '--lr-type', type=str, default=LR_TYPE)

    parser.add_argument('-ll', '--lambda-laplacian',
                        type=float, default=LAMBDA_LAPLACIAN)
    parser.add_argument('-lf', '--lambda-flatten',
                        type=float, default=LAMBDA_FLATTEN)
    parser.add_argument('-ni', '--num-iterations',
                        type=int, default=NUM_ITERATIONS)
    parser.add_argument('-nic', '--num-iterations-classifier',
                        type=int, default=NUM_ITERATIONS_CLASSIFIER)
    parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
    parser.add_argument('-df', '--demo-freq', type=int, default=DEMO_FREQ)
    parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--fisher_estimation_sample_size', type=int, default=1024)
    parser.add_argument('--k', type=int, default=3, help="Number of classes for each task in continual learning.")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    date_time = datetime.now().strftime("%Y-%m-%d-%h-%M")

    # train output directories
    directory_output = os.path.join(args.model_directory, args.experiment_id, date_time)
    os.makedirs(directory_output, exist_ok=True)
    image_output = os.path.join(directory_output, 'pic')
    os.makedirs(image_output, exist_ok=True)

    # test output directories
    directory_mesh = os.path.join(directory_output, 'test')
    os.makedirs(directory_mesh, exist_ok=True)
    num_set = len(datasets.class_ids_map) // args.k
    class_ids = args.class_ids.split(',')

    logging.basicConfig(filename=os.path.join(directory_output, "continual-learning.log"),
                        level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # exclude one class to make 4 sets of 3 classes
    class_ids.pop()

    train_ids = val_ids = [class_ids.pop() for i in range(args.k)]
    state_dict = None

    # holds the initial weights of the classification head
    initial_output_weights = None

    number_task = 1

    while num_set:

        # setup model & optimizer
        model = models.Model('data/obj/sphere/sphere_642.obj', args=args, num_classes=len(val_ids))
        model = model.to(args.device)

        start_iter = START_ITERATION

        if initial_output_weights is None:
            initial_output_weights = (model.mvcnn.classifier[6].weight.detach().clone(),
                                      model.mvcnn.classifier[6].bias.detach().clone())
        if state_dict:
            state_dict["mvcnn.classifier.6.weight"] = torch.cat(
                [state_dict["mvcnn.classifier.6.weight"], initial_output_weights[0]], dim=0)
            state_dict["mvcnn.classifier.6.bias"] = torch.cat(
                [state_dict["mvcnn.classifier.6.bias"], initial_output_weights[1]], dim=0)
            model.load_state_dict(state_dict)
        optimizer = torch.optim.Adam(model.model_param(), args.learning_rate)

        if args.resume_path and number_task == 1:
            state_dicts = torch.load(args.resume_path)
            model.load_state_dict(state_dicts['model'])
            optimizer.load_state_dict(state_dicts['optimizer'])
            start_iter = int(os.path.split(args.resume_path)[1][11:].split('.')[0]) + 1
            logger.info('Resuming from %s iteration for task %d' % (start_iter, number_task))

        # display current classes that we are training/validating on
        logger.info(
            f"Training on {train_ids} which correspond to {[datasets.class_ids_map.get(train_id) for train_id in train_ids]}")
        logger.info(
            f"Validating on {val_ids} which correspond to {[datasets.class_ids_map.get(val_id) for val_id in val_ids]}")

        model.train()
        dataset_train = datasets.ShapeNet(
            args.dataset_directory, train_ids, 'train')
        train(dataset_train=dataset_train, model=model,
              optimizer=optimizer, directory_output=directory_output, image_output=image_output, start_iter=start_iter, args=args)
        # model.consolidate(model.estimate_fisher(
        #     dataset_train, args.fisher_estimation_sample_size
        # ))

        # delete train dataset to free memory
        del dataset_train

        model.eval()
        dataset_val = datasets.ShapeNet(args.dataset_directory, val_ids, 'val')
        test(dataset_val, model, directory_mesh=directory_mesh)

        num_set -= 1
        if num_set:
            train_ids = [class_ids.pop() for i in range(args.k)]
            val_ids.extend(train_ids)

        # store model weights
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(directory_output, "{}.pt".format(str(number_task))))
        number_task += 1
