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

from torch.utils.data import DataLoader

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

NUM_WORKERS = 24
BATCH_SIZE = 64
BATCH_SIZE_CLASSIFIER = 8
LEARNING_RATE = 1e-4
LEARNING_RATE_CLASSIFIER = 5e-5
LR_TYPE = 'step'
NUM_ITERATIONS = 250000
NUM_EPOCHS_CLASSIFIER = 5 # TODO combat overfitting... 

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


def train(dataset_train, model, optimizer, directory_output, image_output, start_iter, args, dataloader):
    end = time.time()
    batch_time = AverageMeter()
    epoch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for i in range(start_iter, args.num_iterations + 1):
        # adjust learning rate and sigma_val (decay after 150k iter)
        lr = adjust_learning_rate(
            [optimizer], args.learning_rate, i, method=args.lr_type)
        model.set_sigma(adjust_sigma(args.sigma_val, i))

        # load images from multi-view
        images_a, images_b, viewpoints_a, viewpoints_b = [
            tensor.to(args.device) for tensor in dataset_train.get_random_batch(args.batch_size)]

        # soft render images
        render_images, laplacian_loss, flatten_loss, _ = model(images=[images_a, images_b],
                                                               viewpoints=[viewpoints_a,
                                                                           viewpoints_b],
                                                               task='train')
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()
        ewc_loss = model.ewc_loss(device=args.device, classifier=False)

        # compute loss
        loss = multiview_iou_loss(render_images, images_a, images_b) + \
            args.lambda_laplacian * laplacian_loss + \
            args.lambda_flatten * flatten_loss + ewc_loss

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
                'iter': i
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
            log_str = 'Iter: [{0}/{1}]\tTime {batch_time.val:.3f}\tLoss {loss.val:.3f}\tlr {lr:.6f}\tsv {sv:.6f}\tewc {ewc:.6f}'.format(i, args.num_iterations,
                                                                                                                           batch_time=batch_time, loss=losses, lr=lr, sv=model.rasterizer.sigma_val,ewc=ewc_loss.item())
            logger.info(log_str)

    # 2nd traning phase which focus on training the 3D shape classifier
    # loss function
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    epoch_end = time.time()
    
    for i in range(1, args.num_epochs_classifier + 1):
        accuracies = []
        losses = []
        for batch_idx, data in enumerate(dataloader, start=1):

            images, viewpoints, labels = [tensor.to(args.device) for tensor in data]
            images = images.view(-1, images.shape[-3],images.shape[-2],images.shape[-1]).contiguous()
            viewpoints = viewpoints.view(-1, viewpoints.shape[-1]).contiguous()

            with torch.no_grad():
                silhouettes = model(images=images,
                    viewpoints=viewpoints, task="classify")
            
            # compute classification loss
            rendered_images = silhouettes.detach().clone()
            rendered_images.requires_grad = True

            # only taking first three channels as the 4th channel is a mask
            class_predictions = model.mvcnn(rendered_images[:, :3, :, :])

            # compute cross entropy loss
            ce_loss = classification_loss_fn(class_predictions, labels)
            ewc_loss = model.ewc_loss(device=args.device, classifier=True)

            # accuracy
            _max_values, max_indices = torch.max(class_predictions, dim=1)
            batch_acc = torch.sum(max_indices == labels) / args.batch_size_classifier
            loss = ce_loss + ewc_loss

            accuracies.append(batch_acc.item())
            losses.append(loss.item())
            # compute gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()
            logger.info('Batch {0}\tTime: {batch_time.val:.3f}\tCE Loss: {ce_loss:.3f}\tBatch accuracy: {acc:.3f}\tEWC: {ewc:.3f}'.format(batch_idx, batch_time=batch_time, ce_loss=ce_loss.item(), acc=batch_acc.item(), ewc=ewc_loss.item()))

        # end of epoch
        epoch_time.update(time.time() - epoch_end)
        epoch_end = time.time()
        log_str = 'Epoch: [{0}/{1}]\tEpoch time: {epoch_time.val:.3f}\tAverage CE Loss: {ce_loss:.3f}\tEpoch accuracy: {acc:.3f}'.format(i, args.num_epochs_classifier,
                                                                                                                    epoch_time=epoch_time, ce_loss=sum(losses)/len(losses), acc=sum(accuracies)/len(accuracies))
        logger.info(log_str)

    if args.consolidate:
        logger.info("=> Estimating diagonals of the fisher information matrix...")
        print_memory()
        fisher_matrix = model.estimate_fisher(args=args, data_loader=train_dataloader)
        model.consolidate(fisher_matrix)


def validate(dataset_val, model, directory_mesh, args):
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    iou_all = []

    for class_id, class_name in dataset_val.class_ids_pair:

        directory_mesh_cls = os.path.join(directory_mesh, class_id)
        os.makedirs(directory_mesh_cls, exist_ok=True)
        iou = 0
        acc = []
        ce_losses = []

        for i, (im, vx, viewpoints) in enumerate(dataset_val.get_all_batches_for_evaluation(args.batch_size_classifier, class_id)):
            images = torch.autograd.Variable(im).to(args.device)
            voxels = vx.numpy()

            with torch.no_grad():
                batch_iou, vertices, faces, silhouettes = model(
                    images, voxels=voxels, viewpoints=viewpoints, task='test')
                iou += batch_iou.sum()

                # only taking first three channels as the 4th channel is a mask
                class_predictions = model.mvcnn(silhouettes[:, :3, :, :])

            labels = torch.empty(class_predictions.shape[0], dtype=torch.long).fill_(dataset_val.class_ids_labels.get(class_id)).to(args.device)

            # compute cross entropy loss
            ce_loss = classification_loss_fn(class_predictions, labels)

            # accuracy
            _max_values, max_indices = torch.max(class_predictions, dim=1)
            batch_acc = torch.sum(max_indices == labels) / args.batch_size_classifier
            acc.append(batch_acc.item())
            ce_losses.append(ce_loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            # save demo images
            for k in range(vertices.size(0)):
                obj_id = (i * args.batch_size_classifier + k)
                if obj_id % args.save_freq == 0:
                    mesh_path = os.path.join(
                        directory_mesh_cls, '%06d.obj' % obj_id)
                    input_path = os.path.join(
                        directory_mesh_cls, '%06d.png' % obj_id)
                    srf.save_obj(mesh_path, vertices[k], faces[k])
                    imageio.imsave(input_path, img_cvt(images[k]))

        iou_cls = iou / 24. / dataset_val.num_data[class_id] * 100
        iou_all.append(iou_cls)
        logger.info('=================================')
        logger.info('Mean IoU: %.3f for class %s' % (iou_cls, class_name))
        logger.info('Mean Accuracy: %.3f for class %s' % (sum(acc)/len(acc), class_name))
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
        # only decay learning rate for rasterizer
        optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_sigma(sigma, i):
    decay = 0.3
    if i >= 150000:
        sigma *= decay
    return sigma

def print_memory():
    logger.info(f"Allocated: {torch.cuda.memory_allocated(0)}B\tReserverd: {torch.cuda.memory_reserved(0)}B\tTotal memory: {torch.cuda.get_device_properties(0).total_memory}B")

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
    parser.add_argument('-bc', '--batch-size-classifier', type=int, default=BATCH_SIZE_CLASSIFIER)
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
    parser.add_argument('-nic', '--num-epochs-classifier',
                        type=int, default=NUM_EPOCHS_CLASSIFIER)
    parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
    parser.add_argument('-df', '--demo-freq', type=int, default=DEMO_FREQ)
    parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--fisher_estimation_sample_size', type=int, default=1024)
    parser.add_argument('--k', type=int, default=3, help="Number of classes for each task in continual learning.")
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS, help="Number of workers for dataloaders.")
    parser.add_argument('--consolidate', action='store_true', help="Use of EWC loss to consolidate neural network.")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

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

    task_number = 1

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
        optimizer = torch.optim.Adam(model.model_param(args.learning_rate_classifier), args.learning_rate)
        # if args.resume_path and task_number == 1:
        #     state_dicts = torch.load(args.resume_path)
        #     model.load_state_dict(state_dicts['model'])
        #     # optimizer.load_state_dict(state_dicts['optimizer'])
        #     # start_iter = state_dicts['iter']
        #     logger.info('Resuming from %s iteration for task %d' % (start_iter, task_number))
        #     del state_dicts
        #     torch.cuda.empty_cache()

        # display current classes that we are training/validating on
        logger.info(
            f"Training on {train_ids} which correspond to {[datasets.class_ids_map.get(train_id) for train_id in train_ids]}")
        logger.info(
            f"Validating on {val_ids} which correspond to {[datasets.class_ids_map.get(val_id) for val_id in val_ids]}")
        model.train()
        dataset_train = datasets.ShapeNet(
            args.dataset_directory, train_ids, 'train')
        train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size_classifier, shuffle=True, num_workers=args.num_workers)
        task_directory_output = os.path.join(directory_output, str(task_number))
        os.makedirs(task_directory_output, exist_ok=True)
        task_image_output = os.path.join(image_output, str(task_number))
        os.makedirs(task_image_output, exist_ok=True)
        train(dataset_train=dataset_train, model=model,
              optimizer=optimizer, directory_output=task_directory_output, image_output=task_image_output, start_iter=start_iter, args=args, dataloader=train_dataloader)

        # delete train dataset to free memory
        del dataset_train

        model.eval()
        dataset_val = datasets.ShapeNet(args.dataset_directory, val_ids, 'val')
        task_directory_mesh = os.path.join(directory_mesh, str(task_number))
        os.makedirs(task_directory_mesh, exist_ok=True)
        validate(dataset_val, model, directory_mesh=task_directory_mesh, args=args)

        # store model weights
        state_dict = model.state_dict()
        torch.save({
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'iter': 0,
                'task_number': task_number,
                'train_ids': train_ids,
                'val_ids': val_ids
            }, os.path.join(directory_output, "{}.pt".format(str(task_number))))
        
        num_set -= 1
        if num_set:
            train_ids = [class_ids.pop() for _ in range(args.k)]
            val_ids.extend(train_ids)

        task_number += 1