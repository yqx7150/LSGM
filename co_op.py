import torch
def get_mask(image):
    mask = torch.cat([torch.ones_like(image[:, :1, ...]),
                      torch.zeros_like(image[:, 1:, ...])], dim=1)
    return mask
  

# Decouple a gray-scale image with `M`
def decouple(inputs):
  # `M` is an orthonormal matrix to decouple image space to a latent space where the gray-scale image
  # occupies a separate channel
  M = torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                 [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                 [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
  # `invM` is the inverse transformation of `M`
  invM = torch.inverse(M)
  return torch.einsum('bihw,ij->bjhw', inputs, M.to(inputs.device))

# The inverse function to `decouple`.
def couple(inputs):
  # `M` is an orthonormal matrix to decouple image space to a latent space where the gray-scale image
  # occupies a separate channel
  M = torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                 [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                 [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
  # `invM` is the inverse transformation of `M`
  invM = torch.inverse(M)
  return torch.einsum('bihw,ij->bjhw', inputs, invM.to(inputs.device))
