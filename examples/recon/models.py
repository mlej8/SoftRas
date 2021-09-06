import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from torchvision import models, transforms
import soft_renderer as sr
import soft_renderer.functional as srf
import math


class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        return x


class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])  # vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])  # faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces


class MVCNN(nn.Module):
    """Implementation of Multi-view Convolutional Neural Networks for 3D Shape Recognition from https://arxiv.org/abs/1505.00880"""

    def __init__(self, num_classes, num_views, pretrained):
        super(MVCNN, self).__init__()
        self.num_views = num_views
        model = models.vgg11(pretrained=pretrained)
        self.feature_extractor = model.features
        self.classifier = model.classifier
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)
        self.transform = transforms.Resize((224, 224))

    def forward(self, images):
        # resize the images
        images = self.transform(images)

        output = self.feature_extractor(images)
        output = output.view(images.shape[0]//self.num_views, self.num_views,
                             output.shape[-3], output.shape[-2], output.shape[-1])

        # view pooling layer (pooling along the view dimension - maximum operation across the views)
        max_values, _indices = torch.max(output, dim=1)

        return self.classifier(max_values.view(output.shape[0], -1))


class Model(nn.Module):
    def __init__(self, filename_obj, args, num_classes, num_views=24, lamda=40, pretrained=True):
        super(Model, self).__init__()

        # lambda for ewc loss which sets how important the old task is compared with the new one
        self.lamda = lamda

        # auto-encoder
        self.encoder = Encoder(im_size=args.image_size)
        self.decoder = Decoder(filename_obj)

        # renderer
        self.transform = sr.LookAt(viewing_angle=15)
        self.lighting = sr.Lighting()
        self.rasterizer = sr.SoftRasterizer(image_size=args.image_size, sigma_val=args.sigma_val,
                                            aggr_func_rgb='hard', aggr_func_alpha='prod', dist_eps=1e-10)

        # mesh regularizer
        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)

        # classifier
        self.mvcnn = MVCNN(num_classes, num_views, pretrained)

    def model_param(self, learning_rate_classifier):
        return [{"params": list(self.encoder.parameters()) + list(self.decoder.parameters())}, {"params": list(self.mvcnn.parameters()), 'lr': learning_rate_classifier}]

    def set_sigma(self, sigma):
        self.rasterizer.set_sigma(sigma)

    def reconstruct(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces

    def render_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b):
        # [Ia, Ib]
        images = torch.cat((image_a, image_b), dim=0)  # torch.Size([64, 4, 64, 64])
        # [Va, Va, Vb, Vb], set viewpoints
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)  # torch.Size([128, 3])
        self.transform.set_eyes(viewpoints)

        vertices, faces = self.reconstruct(images)
        laplacian_loss = self.laplacian_loss(vertices)
        flatten_loss = self.flatten_loss(vertices)

        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices), dim=0)  # torch.Size([128, 642, 3])
        faces = torch.cat((faces, faces), dim=0)

        # [Raa, Rba, Rab, Rbb], render for cross-view consistency
        mesh = sr.Mesh(vertices, faces)
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        silhouettes = self.rasterizer(mesh)
        render_images = silhouettes.chunk(4, dim=0)

        return render_images, laplacian_loss, flatten_loss, silhouettes

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.reconstruct(images)

        # computing IOU
        faces_ = srf.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))

        return iou, vertices, faces

    def forward(self, images=None, viewpoints=None, voxels=None, task='train'):
        if task == 'train':
            return self.render_multiview(images[0], images[1], viewpoints[0], viewpoints[1])
        elif task == "classify":
            assert images is not None, "ground truth images must be provided for classification."
            assert viewpoints is not None, "viewpoints must be provided."

            self.transform.set_eyes(viewpoints)
            vertices, faces = self.reconstruct(images)
            mesh = sr.Mesh(vertices, faces)
            mesh = self.lighting(mesh)
            mesh = self.transform(mesh)
            silhouettes = self.rasterizer(mesh)
            return silhouettes

        elif task == 'test':
            assert images is not None, "ground truth images must be provided for classification."
            assert viewpoints is not None, "viewpoints must be provided."
            assert voxels is not None, "voxels must be provided for evaluation."
            iou, vertices, faces = self.evaluate_iou(images, voxels)

            self.transform.set_eyes(viewpoints)
            mesh = sr.Mesh(vertices, faces)
            mesh = self.lighting(mesh)
            mesh = self.transform(mesh)
            silhouettes = self.rasterizer(mesh)
            return iou, vertices, faces, silhouettes

    def ewc_loss(self, device, classifier):
        try:
            losses = []
            for n, p in self.named_parameters():
                network_name = n.split(".")[0]
                if classifier and network_name != "mvcnn":
                    continue
                elif not classifier and network_name == "mvcnn":
                    continue
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = torch.tensor(getattr(self, '{}_mean'.format(n)))
                fisher = torch.tensor(getattr(self, '{}_fisher'.format(n)))
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return torch.zeros(1).to(device)

    def estimate_fisher(self, data_loader, args):
        # sample loglikelihoods from the dataset.
        loglikelihoods_grads = []
        for i, data in enumerate(data_loader, 1):
            print(f"batch {i} Allocated: {torch.cuda.memory_allocated(0)}B\tReserverd: {torch.cuda.memory_reserved(0)}B\tTotal memory: {torch.cuda.get_device_properties(0).total_memory}B")
            
            images, viewpoints, labels = [tensor.to(args.device) for tensor in data]
            images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]).contiguous()
            viewpoints = viewpoints.view(-1, viewpoints.shape[-1]).contiguous()

            silhouettes = self(images=images,
                                viewpoints=viewpoints, task="classify")

            # only taking first three channels as the 4th channel is a mask
            class_predictions = self.mvcnn(silhouettes[:, :3, :, :])
            torch.cuda.synchronize()
            loglikelihoods = F.log_softmax(class_predictions, dim=1)[range(args.batch_size_classifier), labels.data].cpu().unbind()
            for j, l in enumerate(loglikelihoods, 1):
                gradients = autograd.grad(l, self.parameters(), retain_graph=(j < len(loglikelihoods)))
                torch.cuda.synchronize()
                print(f"{j}Allocated: {torch.cuda.memory_allocated(0)}B\tReserverd: {torch.cuda.memory_reserved(0)}B\tTotal memory: {torch.cuda.get_device_properties(0).total_memory}B")
                
                loglikelihoods_grads.append(tuple(g.cpu() for g in gradients))
                print(f"{j}Allocated: {torch.cuda.memory_allocated(0)}B\tReserverd: {torch.cuda.memory_reserved(0)}B\tTotal memory: {torch.cuda.get_device_properties(0).total_memory}B")

                del gradients
                torch.cuda.empty_cache()
                print(f"{j}Allocated: {torch.cuda.memory_allocated(0)}B\tReserverd: {torch.cuda.memory_reserved(0)}B\tTotal memory: {torch.cuda.get_device_properties(0).total_memory}B")

            del silhouettes, class_predictions, images, viewpoints, labels, loglikelihoods
            torch.cuda.empty_cache()
            if i >= args.fisher_estimation_sample_size // args.batch_size_classifier:
                break
                
        # estimate the fisher information of the parameters.
        loglikelihood_grads = zip(*loglikelihoods_grads)
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher_matrix):
        for name, p in self.named_parameters():
            name = name.replace('.', '__')
            self.register_buffer(f'{name}_mean', p.data.clone())
            self.register_buffer(f'{name}_fisher', fisher_matrix[name].data.clone())

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
