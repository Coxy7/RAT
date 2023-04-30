import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks.attack import Attack


class SoftPGD(Attack):

    def __init__(self, model, teacher, eps=0.3, alpha=2/255, steps=40, random_start=False):
        super().__init__("SoftPGD", model)
        self.teacher = teacher
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        with torch.no_grad():
            logits_teacher = self.teacher(images)
        
        # loss = nn.KLDivLoss(size_average=False)
        loss = nn.KLDivLoss(reduction='batchmean')

        adv_images = images.clone().detach()

        if self.random_start:
            # adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            # adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            adv_images = adv_images + 1e-3 *  torch.empty_like(adv_images).normal_() # TRADES original

        for i in range(self.steps):
            adv_images.requires_grad = True

            cost = loss(F.log_softmax(self.model(adv_images), dim=1),
                        F.softmax(logits_teacher, dim=1))

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
