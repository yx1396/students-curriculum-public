import torch


class EMA:
    """参数的指数滑动平均：shadow = decay*shadow + (1-decay)*param。"""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        """更新影子参数（指数滑动平均）。"""
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        """将影子参数复制回模型参数。"""
        for n, p in model.named_parameters():
            p.copy_(self.shadow[n])
