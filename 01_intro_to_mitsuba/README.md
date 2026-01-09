# Assignment 1: Introduction to Variance Reduction for Physically-Based Rendering & Mitsuba 3

## Goal

This assignment aims to familiarize you with physically-based rendering and advanced Monte Carlo techniques, as well as the Mitsuba 3 framework. You will gain hands-on experience with physically-based rendering concepts and learn to implement a basic path-tracer.

## Reading

1. [PBRT Chapter 2, Sections 1-4](https://www.pbr-book.org/4ed/Monte_Carlo_Integration)
2. [PBRT Chapter 8, Section 1](https://www.pbr-book.org/4ed/Sampling_and_Reconstruction)
3. [PBRT Chapter 13, Sections 1-4 & Further Reading](https://www.pbr-book.org/4ed/Light_Transport_I_Surface_Reflection)

## Written assignment

Work through the following exercise sets on Monte Carlo integration techniques:

- [Multiple Importance Sampling Exercises](mis_exercises.pdf)
- [Control Variate Exercises](control_variate_exercises.pdf)

You are not required to complete all exercises, but make an attempt at them and try to understand the underlying concepts. These exercises will strengthen your understanding of variance reduction techniques used in Monte Carlo renderers.

## Project

### 1. Mitsuba 3 Environment Setup

Install Mitsuba 3 following the [official installation guide](https://mitsuba.readthedocs.io/en/stable/#). We recommend using pip:

```bash
pip install mitsuba
```

Verify your installation:

```python
import mitsuba as mi
mi.set_variant('scalar_rgb')
print(mi.__version__)
```

**Available variants**: Mitsuba 3 provides different variants optimized for different use cases:
- `scalar_rgb`: Single-ray processing, simple but **very slow** for custom Python integrators
- `llvm_ad_rgb`: Vectorized CPU processing with autodiff support
- `cuda_ad_rgb`: GPU-accelerated processing with autodiff support (recommended)

### 2. Rendering a Cornell Box Scene

Render the provided Cornell box scene using Mitsuba's Python API:

```python
import mitsuba as mi
mi.set_variant('scalar_rgb')

# Load and render the scene
scene = mi.load_file('cornell_box.xml')
image = mi.render(scene, spp=64)

# Save the result
mi.util.write_bitmap('output.png', image)
```

Experiment with different sample counts (`spp`) and observe the effect on image quality and rendering time.

### 3. Creating a New Integrator

Create a custom integrator by subclassing Mitsuba's base integrator class. Start with a simple direct lighting integrator.

```python
import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')  # Or 'llvm_ad_rgb' for CPU

class DirectIntegrator(mi.SamplingIntegrator):
    """Direct lighting integrator with same sample consumption as PathTracerNEE's first bounce."""

    def __init__(self, props=mi.Properties()):
        super().__init__(props)

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        result = mi.Color3f(0.0)
        ray = mi.Ray3f(ray)
        active = mi.Bool(active)

        # Ray intersection
        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()

        ctx = mi.BSDFContext()
        bsdf = si.bsdf()

        # Emitted light (direct hit on emitter)
        emitter = si.emitter(scene)
        emitter_contrib = dr.select(
            emitter != None,
            emitter.eval(si, active),
            mi.Color3f(0.0)
        )
        result[active] += emitter_contrib

        # NEE - same sample consumption as PathTracerNEE
        ds, emitter_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active
        )

        active_light = active & (ds.pdf > 0)
        wo_light = si.to_local(ds.d)
        bsdf_val = bsdf.eval(ctx, si, wo_light, active_light)
        visible = ~scene.ray_test(si.spawn_ray_to(ds.p), active_light)
        nee_contrib = emitter_weight * bsdf_val * dr.abs(dr.dot(si.n, ds.d))
        result[active_light & visible] += nee_contrib

        return result, mi.Bool(True), []

# Register the integrator
mi.register_integrator("direct", lambda props: DirectIntegrator(props))
```

#### Using Your Custom Integrator

After registering your custom integrator, there are two ways to use it when rendering:

**Option 1: Pass the integrator directly to `mi.render()`**

```python
# Create an instance of your custom integrator
my_integrator = DirectIntegrator()

# Load the scene and render with your integrator
scene = mi.load_file('cornell_box.xml')
image = mi.render(scene, integrator=my_integrator, spp=64)
mi.util.write_bitmap('direct_lighting.png', image)
```

**Option 2: Load the integrator by its registered name**

```python
# Load integrator from the registry using mi.load_dict()
my_integrator = mi.load_dict({'type': 'direct'})

# Render with the loaded integrator
scene = mi.load_file('cornell_box.xml')
image = mi.render(scene, integrator=my_integrator, spp=64)
```

### 4. Implementing a Basic Path Tracer

Extend your integrator to implement a simple path tracer with a fixed maximum depth and Next Event Estimation (NEE). For vectorized variants (`cuda_ad_rgb` or `llvm_ad_rgb`), use the `@dr.syntax` decorator and `while dr.hint()` pattern for efficient looping.

**Algorithm Overview:**

1. **Initialize** path variables:
   - Set throughput = (1, 1, 1) (RGB color representing path weight)
   - Set result = (0, 0, 0) (accumulated radiance)
   - Set depth = 0

2. **Path tracing loop** (continue while depth < max_depth and rays are active):

   a. **Ray intersection**: Intersect the current ray with the scene
   c. **Next Event Estimation (NEE)**: Sample direct illumination from lights
   d. **Hemisphere sampling**: Sample a direction for the next bounce
   e. **Update** throughput, ray, and depth

3. **Return** the accumulated radiance

**Implementation hints:**

```python
class PathTracerNEE(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        # Initialize: throughput, result, depth, ray, active, ctx
        # ...

        while dr.hint(active, max_iterations=self.max_depth, label="Path Tracer"):
            # 1. Ray intersection
            # 3. NEE: sample lights and add direct illumination
            # 4. Sample next direction using cosine hemisphere
            # 5. Update throughput and spawn next ray
            pass

        return result, mi.Bool(True), []
```

**Useful functions:**

- `mi.warp.square_to_cosine_hemisphere(sample_2d)`: Maps 2D uniform sample → hemisphere direction (local coords)
- `mi.warp.square_to_cosine_hemisphere_pdf(wo)`: Returns PDF = cos(θ) / π
- `si.to_world(wo_local)`: Convert local direction to world coordinates
- `si.spawn_ray(direction)`: Create ray from surface point

**Vectorized loop pattern:**

```python
@dr.syntax
def sample(self, ...):
    active = mi.Bool(active)
    depth = mi.UInt32(0)

    while dr.hint(active, max_iterations=self.max_depth, label="..."):
        # Use boolean masking instead of if statements
        active &= si.is_valid()
        result[active] += contribution
        depth += 1
        active &= depth < self.max_depth
```

**Key formulas:**

- **Rendering equation**:
  ```
  L(x, ω_o) = L_e(x, ω_o) + ∫_Ω f_r(x, ω_i, ω_o) L_i(x, ω_i) |cos(θ_i)| dω_i
  ```

- **Monte Carlo estimator**:
  ```
  L ≈ L_e + [f_r × L_i × |cos(θ)|] / pdf
  ```

- **Throughput update**: `throughput *= f_r(ω_i, ω_o) / pdf(ω_o)`

### 5. Multiple Importance Sampling Comparison

Implement two versions of your path tracer: one with cosine hemisphere sampling (from Section 4) and one with MIS. The MIS version combines **cosine hemisphere sampling** and **BSDF sampling** for the indirect bounces, which significantly improves convergence for glossy materials.

**Power heuristic for MIS:**
```
w_a(x) = [pdf_a(x)]^β / ([pdf_a(x)]^β + [pdf_b(x)]^β)
```
where β = 2 is typically used.

**Implementation approach:**

At each bounce, sample **both** strategies and weight their contributions:

1. **Cosine hemisphere sample**: Draw direction, compute its contribution weighted by MIS
2. **BSDF sample**: Draw direction, compute its contribution weighted by MIS
3. **Combine**: Add both weighted contributions to the throughput

```python
def power_heuristic(pdf_a, pdf_b):
    """Compute MIS weight using power heuristic (beta=2)"""
    a = dr.square(pdf_a)
    b = dr.square(pdf_b)
    return dr.select(pdf_a > 0, a / (a + b), 0.0)
```

**Key insight:** For each strategy, you need to evaluate the PDF of *both* strategies for the sampled direction:
- Cosine sample at direction ω: weight = power_heuristic(cos_pdf(ω), bsdf_pdf(ω))
- BSDF sample at direction ω: weight = power_heuristic(bsdf_pdf(ω), cos_pdf(ω))

**Useful functions:**
- `bsdf.sample(ctx, si, sample_1d, sample_2d, active)`: Returns `(bsdf_sample, bsdf_weight)`
- `bsdf.pdf(ctx, si, wo, active)`: Returns PDF for a given direction

**Key differences from the cosine-only integrator:**
- Both cosine hemisphere and BSDF sampling are performed at each bounce
- Each strategy's contribution is weighted by the power heuristic
- The combined throughput provides lower variance, especially for glossy materials

**Comparison tasks:**
- Render scenes with both versions (with and without MIS) and compare
- Test scenes with small, bright light sources (where MIS helps significantly)
- Test scenes with glossy materials (see below for how to modify materials)
- Show a visible comparison between techniques

**Modifying materials in the XML file:**

To test with glossy materials, modify the BSDF definitions in `cornell_box.xml`. Replace a diffuse BSDF with a glossy conductor or roughplastic BSDF:

```xml
<!-- Original diffuse BSDF -->
<bsdf type="diffuse" id="box">
    <spectrum name="reflectance" value="400:0.343, 404:0.445, ..."/>
</bsdf>

<!-- Replace with rough conductor (glossy metal) -->
<bsdf type="roughconductor" id="box">
    <string name="material" value="Al"/>  <!-- Aluminum -->
    <float name="alpha" value="0.1"/>      <!-- Roughness: 0=mirror, 1=diffuse -->
</bsdf>

<!-- Or replace with rough plastic (glossy dielectric) -->
<bsdf type="roughplastic" id="box">
    <spectrum name="diffuse_reflectance" value="0.3, 0.3, 0.8"/>  <!-- Blue tint -->
    <float name="alpha" value="0.05"/>  <!-- Low roughness = more glossy -->
</bsdf>
```

Available conductor materials: `Al` (aluminum), `Au` (gold), `Cu` (copper), `Ag` (silver)

The `alpha` parameter controls glossiness:
- `alpha = 0.0`: Perfect mirror (ideal specular)
- `alpha = 0.05`: Very glossy (sharp highlights)
- `alpha = 0.2`: Moderately glossy
- `alpha = 1.0`: Nearly diffuse

### 6. Control Variate Rendering

Implement control variate variance reduction. Control variates are an **unbiased** variance reduction technique where the variance reduction depends on the **correlation** between the control variate and the target integral:
- High correlation → significant variance reduction
- Low correlation → little benefit, possibly increased variance

**Control Variate Formula:**

```
L_cv = L_target_low + alpha * (E[L_control] - L_control_low)
```

Where:
- `L_target_low`: Low-sample estimate of the target (e.g., path tracing)
- `L_control_low`: Low-sample estimate of the control variate (using the **same samples**)
- `E[L_control]`: Expected value of the control variate (high-sample estimate)
- `alpha`: Balance coefficient

**Ensuring Sample Correlation**

For control variates to work, `L_target_low` and `L_control_low` must use **correlated random samples**. Use the same `seed` parameter when rendering both to ensure correlation in the first-bounce samples (which dominate the direct lighting contribution).

### 7. Control Variate Analysis

Implement and compare **three control variate approaches** to understand how correlation affects variance reduction:

#### Part A: Perfect Control Variate (Same Integrator)

The **perfect** control variate uses the same path tracing integrator. Since both renders use identical sample sequences (same seed), they are perfectly correlated.

```python
import numpy as np
import matplotlib.pyplot as plt

scene = mi.load_file('cornell_box.xml')
path_integrator = PathTracerNEE()

seed = 12345
low_spp = 8

# Low-sample path tracing (our noisy estimate)
L_path_low = np.array(mi.render(scene, integrator=path_integrator, spp=low_spp, seed=seed))

# Perfect control variate: same integrator, same seed
L_perfect_low = np.array(mi.render(scene, integrator=path_integrator, spp=low_spp, seed=seed))

# Ground truth
ground_truth = np.array(mi.render(scene, integrator=path_integrator, spp=4096))
E_perfect = ground_truth

# With perfect correlation, alpha=1.0 gives optimal variance reduction
for alpha in [0.0, 0.5, 1.0]:
    L_cv = L_path_low + alpha * (E_perfect - L_perfect_low)
    mse = np.mean((L_cv - ground_truth) ** 2)
    print(f"Perfect CV - Alpha: {alpha:.2f}, MSE: {mse:.6f}")
```

#### Part B: Good Control Variate (Direct Lighting)

Direct illumination is **highly correlated** with global illumination because they share the same first-bounce samples. This makes it an excellent practical control variate.

```python
direct_integrator = DirectIntegrator()

# Direct lighting with same seed (correlated first bounce)
L_direct_low = np.array(mi.render(scene, integrator=direct_integrator, spp=low_spp, seed=seed))
E_direct = np.array(mi.render(scene, integrator=direct_integrator, spp=1024))

for alpha in [0.0, 0.5, 1.0]:
    L_cv = L_path_low + alpha * (E_direct - L_direct_low)
    mse = np.mean((L_cv - ground_truth) ** 2)
    print(f"Direct CV - Alpha: {alpha:.2f}, MSE: {mse:.6f}")
```

#### Part C: So-So Control Variate (Scaled Scene)

To understand how **poor correlation** affects control variates, use a **geometrically modified scene** (boxes scaled by 20%). 

```python
def create_scaled_scene(scale=1.20):
    """Create scene with scaled box geometry."""
    # Load base scene and modify box transforms
    # Scale boxes around their centers by the given factor
    ...

scaled_scene = create_scaled_scene(scale=1.20)

# Scaled scene with same seed
L_scaled_low = np.array(mi.render(scaled_scene, integrator=path_integrator, spp=low_spp, seed=seed))
E_scaled = np.array(mi.render(scaled_scene, integrator=path_integrator, spp=1024))

for alpha in [0.0, 0.5, 1.0]:
    L_cv = L_path_low + alpha * (E_scaled - L_scaled_low)
    mse = np.mean((L_cv - ground_truth) ** 2)
    print(f"Scaled CV - Alpha: {alpha:.2f}, MSE: {mse:.6f}")
```

#### Part D: Visualization and Comparison

Create separate comparison images for each control variate type:

```python
# Generate three separate analysis images:
# 1. perfect_cv_analysis.png - Perfect CV results
# 2. direct_cv_analysis.png - Direct lighting CV results
# 3. scaled_cv_analysis.png - Scaled scene CV results

def visualize_cv_results(name, L_low, E_high, L_cv_low, ground_truth, alphas):
    """Create visualization for a control variate approach."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Inputs and CV renders at different alphas
    # Row 2: Error images and MSE plot

    results = []
    for alpha in alphas:
        cv_image = L_low + alpha * (E_high - L_cv_low)
        error = cv_image - ground_truth
        mse = np.mean(error ** 2)
        results.append({'alpha': alpha, 'mse': mse, 'image': cv_image, 'error': error})

    # Plot renders at alpha = 0.0, 0.5, 1.0
    # Plot corresponding error images (amplified)
    # Plot MSE vs alpha curve

    plt.savefig(f'{name}_cv_analysis.png')
```

**Analysis questions:**
- For each of the control variates, where do you see decreased noise? Why?
- For each of the control variates, where do you see increased noise? Why?