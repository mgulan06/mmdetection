import torch
from mmengine.model import BaseModel


class A(BaseModel):
    def __init__(self):
        self.a = 5
    def forward(self, *a, **b):
        return

    def __getattr__(self, key):
        return f"i dont have {key}"

a = A()
print(a.a)
print(a.b)