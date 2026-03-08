# Diffusion Models – Implementation Roadmap

This document tracks the step-by-step improvements from a basic **DDPM implementation** toward modern diffusion systems.

Current status:  
- ✅ Vanilla DDPM implemented  
- ✅ Forward diffusion working  
- ✅ UNet noise prediction working  
- ✅ Reverse sampler working  
- 🔄 Training optimization and improvements remaining  

---

# Phase 1 — Stabilize the Vanilla DDPM

Goal: Make sure the baseline implementation is correct and reproducible.

## TODO

- [ ] Overfit on a single image
- [ ] Verify reverse process reconstructs image
- [ ] Visualize intermediate steps of denoising
- [ ] Plot ε prediction error vs timestep
- [ ] Validate forward diffusion noise statistics
- [ ] Implement deterministic sampling (no stochastic noise)

## Resources

Paper  
- https://arxiv.org/abs/2006.11239  
  *Denoising Diffusion Probabilistic Models*

Blogs  
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/  
- https://huggingface.co/blog/annotated-diffusion  

Video  
- https://www.youtube.com/watch?v=HoKDTa5jHvg  
  (Yannic Kilcher DDPM explanation)

---

# Phase 2 — Improved DDPM (major improvements)

Goal: Improve training stability and sample quality.

## TODO

- [ ] Implement **cosine noise schedule**
- [ ] Implement **v-prediction objective**
- [ ] Implement **importance timestep sampling**
- [ ] Track per-timestep training loss

## Resources

Paper  
- https://arxiv.org/abs/2102.09672  
  *Improved Denoising Diffusion Probabilistic Models*

Blog  
- https://huggingface.co/blog/annotated-diffusion#improved-ddpm

Video  
- https://www.youtube.com/watch?v=zc5NTeJbk-k

---

# Phase 3 — Faster Sampling

Goal: Reduce sampling steps from **1000 → 50 or fewer**.

## TODO

- [ ] Implement **DDIM sampling**
- [ ] Implement **deterministic reverse process**
- [ ] Experiment with fewer sampling steps
- [ ] Compare quality vs number of steps

## Resources

Paper  
- https://arxiv.org/abs/2010.02502  
  *Denoising Diffusion Implicit Models*

Blog  
- https://huggingface.co/blog/ddim

Video  
- https://www.youtube.com/watch?v=St1giarCHjY

---

# Phase 4 — Advanced Samplers

Goal: Very fast sampling (10–20 steps).

## TODO

- [ ] Implement **DPM-Solver**
- [ ] Implement **Heun sampler**
- [ ] Implement **Euler sampler**
- [ ] Compare sampler speed and quality

## Resources

Paper  
- https://arxiv.org/abs/2206.00927  
  *DPM-Solver*

Blog  
- https://huggingface.co/blog/what-are-diffusion-models

Video  
- https://www.youtube.com/watch?v=H45lF4sUgiE

---

# Phase 5 — Architecture Improvements

Goal: Improve UNet performance.

## TODO

- [ ] Add **attention blocks**
- [ ] Add **residual blocks**
- [ ] Implement **better timestep embeddings**
- [ ] Add **group normalization**
- [ ] Add **classifier-free guidance**

## Resources

Paper  
- https://arxiv.org/abs/2205.11487  
  *Imagen*

Blog  
- https://jalammar.github.io/illustrated-stable-diffusion/

---

# Phase 6 — Latent Diffusion

Goal: Reduce compute cost dramatically.

Instead of diffusing images directly:
