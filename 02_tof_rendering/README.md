# Assignment 2: Time-of-Flight Rendering

## Goal

This assignment introduces time-of-flight (ToF) rendering concepts and transient light transport. You will extend your path tracer to track path lengths and build a transient renderer that captures how light propagates through a scene over time.

## Reading

1. [Phasor Imaging: A Generalization of Correlation-Based Time-of-Flight Imaging](https://dl.acm.org/doi/10.1145/2735702)
2. [A Monte Carlo Rendering Framework for Simulating Optical Heterodyne Detection](https://juhyeonkim95.github.io/project-pages/ohd_rendering/static/pdfs/OHD_SIGGRAPH_2025.pdf)
3. [A Theory of Fermat Paths for Non-Line-of-Sight Shape Reconstruction](https://imaging.cs.cmu.edu/fermat_paths/assets/cvpr2019.pdf)

## Written Assignment

Work through the exercises in [tof_exercises.pdf](tof_exercises.pdf).

## Project Part 1: Transient Path Tracing

### 1. Setup: Install mitransient

Install the mitransient library which provides helper classes for transient rendering:

```bash
pip install mitransient
```

Verify your installation:

```python
import mitsuba as mi
import mitransient as mitr

mi.set_variant('cuda_ad_rgb')
print(mitr.__version__)
```

### 2. Implementing a Gaussian Pulse

Implement a `GaussianPulse` class that represents a normalized Gaussian temporal pulse. This will be used to model the temporal profile of light sources.

```python
import drjit as dr
import mitsuba as mi
import numpy as np

class GaussianPulse:
    """
    Normalized Gaussian pulse centered at t=0.

    The pulse is defined as:
        p(t) = (1 / (σ √(2π))) × exp(-t² / (2σ²))

    where σ is the standard deviation (width_opl in optical path length units).
    """

    def __init__(self, width_opl: float):
        """
        Initialize the Gaussian pulse.

        Args:
            width_opl: Standard deviation in optical path length units (meters)
        """
        self.width_opl = width_opl
        # TODO: Pre-compute the normalization factor
        self.normalization = ...

    def eval(self, t: mi.Float) -> mi.Float:
        """
        Evaluate the Gaussian at time offset t.

        Args:
            t: Time offset from pulse center (in OPL units)

        Returns:
            Normalized Gaussian value at t
        """
        # TODO: Implement Gaussian evaluation
        pass

    def sample(self, xi: mi.Float):
        """
        Sample a time offset from the Gaussian distribution using inverse CDF.

        Args:
            xi: Uniform random value in [0, 1]

        Returns:
            Tuple of (sampled_time, weight) where weight is always 1.0
            for importance sampling
        """
        # TODO: Implement inverse CDF sampling using the inverse error function
        # Hint: For a Gaussian, the inverse CDF is: μ + σ × √2 × erfinv(2ξ - 1)
        # Use dr.erfinv() for the inverse error function
        pass
```

**Testing your implementation:**

```python
# Test that the pulse integrates to 1
pulse = GaussianPulse(width_opl=0.05)
t = dr.linspace(mi.Float, -0.5, 0.5, 1000)
dt = 1.0 / 1000
integral = dr.sum(pulse.eval(t)) * dt
print(f"Integral: {integral[0]:.4f} (should be ~1.0)")

# Test sampling
samples = [pulse.sample(dr.opaque(mi.Float, np.random.random()))[0] for _ in range(10000)]
print(f"Sample mean: {np.mean(samples):.4f} (should be ~0.0)")
print(f"Sample std: {np.std(samples):.4f} (should be ~{pulse.width_opl:.4f})")
```

### 3. Time-Domain Path Tracing

Modify your path tracer to accept a `target_time` parameter. The integrator traces paths and weights contributions based on how well the path length matches the target time.

**Key insight:** In transient rendering, we track optical path length (OPL) directly. For scenes in air (refractive index n ≈ 1), this equals the sum of segment lengths:

```
OPL = Σ segment_length
```

The relationship between time and path length is: `time = OPL / c` where c is the speed of light.

**Implementation approach:**

The `sample` method takes `target_time` as input and evaluates the pulse at `target_time - path_length`:

```python
class TimeGatedTransientPath(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)
        self.pulse = GaussianPulse(width_opl=props.get('pulse_width_opl', 0.03))

    @dr.syntax
    def sample(self, scene, sampler, ray, target_time, medium=None, active=True):
        """
        Sample paths and weight by temporal pulse.

        Args:
            scene: The scene to render
            sampler: Random number generator
            ray: Camera ray
            target_time: The target time (in OPL units) we're rendering for
            medium: Participating medium (optional)
            active: Active ray mask

        Returns:
            (color, valid, aov) tuple
        """
        result = mi.Color3f(0.0)
        throughput = mi.Color3f(1.0)
        ray = mi.Ray3f(ray)
        active = mi.Bool(active)

        # Track total path length (optical path length)
        path_length = mi.Float(0.0)
        depth = mi.UInt32(0)

        while dr.hint(active, max_iterations=self.max_depth, label="Path"):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()

            # Accumulate path length
            path_length[active] += si.t

            # For NEE contribution:
            # total_opl = path_length + distance_to_light
            # shifted_time = target_time - total_opl
            # pulse_weight = self.pulse.eval(shifted_time)
            # Contribution is weighted by pulse_weight

            # ... your path tracing code from Assignment 1 ...

        return result, mi.Bool(True), []
```

**Render loop:** To render a full transient image, call `sample()` for many different `target_time` values spanning your temporal range:

```python
def render_transient(scene, integrator, film_config, spp=64):
    """Render transient by sampling different target times."""
    num_bins = film_config['temporal_bins']
    start_opl = film_config['start_opl']
    bin_width = film_config['bin_width_opl']

    transient_data = []

    for bin_idx in range(num_bins):
        # Target time for this bin (can add jitter for anti-aliasing)
        target_time = start_opl + (bin_idx + 0.5) * bin_width

        # Render at this target time
        image = render_at_time(scene, integrator, target_time, spp)
        transient_data.append(image)

    return np.stack(transient_data, axis=-1)
```

### 4. Understanding the TransientHDRFilm

The `TransientHDRFilm` from mitransient stores measurements in time bins. Study its interface:

```python
# Create a scene with a transient film
scene_dict = {
    'type': 'scene',
    'integrator': {'type': 'path'},
    'sensor': {
        'type': 'perspective',
        'film': {
            'type': 'transient_hdr_film',
            'width': 256,
            'height': 256,
            'temporal_bins': 512,       # Number of time bins
            'bin_width_opl': 0.01,      # Width per bin in meters
            'start_opl': 2.0,           # Starting optical path length
        },
        # ... rest of sensor config
    },
    # ... rest of scene
}
```

The film provides methods to:
- `prepare()`: Initialize the film for rendering
- `develop()`: Return `(steady_image, transient_data)` where transient_data has shape `(height, width, temporal_bins, channels)`

### 5. Building a Transient Path Tracer

Create a transient path tracer that outputs time-resolved measurements to the `TransientHDRFilm`.

**Key modifications from your standard path tracer:**

1. **Track cumulative path length** at each bounce
2. **Add contributions to transient film** with temporal coordinates
3. **Apply pulse convolution** using your `GaussianPulse` class

**Pseudocode structure:**

```python
class TransientPath(mi.SamplingIntegrator):
    """
    Transient path tracer that outputs to a TransientHDRFilm.

    Unlike TimeGatedTransientPath which renders a single time slice,
    this integrator adds contributions to all relevant time bins
    based on path length.
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)
        pulse_width = props.get('pulse_width_opl', 0.03)
        self.pulse = GaussianPulse(width_opl=pulse_width)

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        # Initialize: throughput, path_length, depth, etc.

        while dr.hint(active, max_iterations=self.max_depth, label="Transient"):
            # 1. Ray intersection
            # 2. Accumulate path_length += si.t
            # 3. For NEE contribution:
            #    - Compute total_path_length = path_length + distance_to_light
            #    - Sample pulse offset: t_offset = pulse.sample(sampler.next_1d())
            #    - effective_opl = total_path_length + t_offset
            #    - Add contribution to transient film at effective_opl
            # 4. BSDF sampling for next bounce

        return result, mi.Bool(True), []
```

### 6. AMCW Rendering

Implement amplitude-modulated continuous-wave (AMCW) ToF rendering. In AMCW systems, the light source and sensor are modulated at frequency ω, and the measurement encodes phase information.

For a path with optical path length `opl`:

```python
# Phase accumulated along the path
c = 299792458.0  # Speed of light (m/s)
omega = 2 * np.pi * frequency  # Angular frequency
phase = omega * (opl / c)

# Phasor components
phasor_real = radiance * dr.cos(phase)
phasor_imag = radiance * dr.sin(phase)
```

**Implementation task:** Create an AMCW integrator that outputs phasor images (real and imaginary channels).

### 7. Visualization

Visualize your transient renders:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_transient(transient_data, start_opl, bin_width_opl):
    """Create an animation of light propagating through the scene."""
    data = np.array(transient_data)

    # Average RGB channels for visualization
    if data.ndim == 4:
        data = np.mean(data, axis=-1)

    fig, ax = plt.subplots()
    vmax = np.percentile(data, 99)
    im = ax.imshow(data[:, :, 0], cmap='hot', vmin=0, vmax=vmax)

    def update(frame):
        im.set_array(data[:, :, frame])
        opl = start_opl + frame * bin_width_opl
        t_ns = opl / 0.3  # Convert to nanoseconds (c ≈ 0.3 m/ns)
        ax.set_title(f't = {t_ns:.2f} ns')
        return [im]

    anim = FuncAnimation(fig, update, frames=data.shape[2], interval=50, blit=True)
    return anim

def plot_pixel_transient(transient_data, x, y, start_opl, bin_width_opl):
    """Plot temporal response at a single pixel."""
    data = np.array(transient_data)
    response = data[y, x, :]

    times = start_opl + np.arange(len(response)) * bin_width_opl
    times_ns = times / 0.3

    plt.figure()
    plt.plot(times_ns, response)
    plt.xlabel('Time (ns)')
    plt.ylabel('Intensity')
    plt.title(f'Transient at pixel ({x}, {y})')
    plt.show()
```

### Part 1 Deliverables

1. **GaussianPulse implementation** with `eval()` and `sample()` methods
2. **TimeGatedTransientPath integrator** that takes `target_time` as input
3. **TransientPath integrator** that renders to a `TransientHDRFilm`
4. **Visualizations:**
   - Transient animation showing light propagation
   - Single-pixel temporal responses at different scene locations
   - Steady-state image (sum over all time bins)
5. **AMCW phasor images** at 2-3 different modulation frequencies
6. **Brief analysis:** Explain how different path lengths correspond to scene features (direct light, reflections, etc.)

---

## Project Part 2: Confocal Scanning and NLOS Rendering

In this part, you will implement efficient rendering for confocal scanning systems and non-line-of-sight (NLOS) imaging. The key insight is that instead of sampling light sources directly, we can treat the laser-illuminated point on the relay wall as a secondary light source.

### 8. Confocal Scanning Setup

In a confocal NLOS system:
- A **laser** illuminates a point on a visible relay wall
- A **sensor** observes the same point (confocal = co-located laser and sensor view)
- Light travels: laser → relay wall → hidden scene → relay wall → sensor
- The transient measurement reveals information about the hidden geometry

```
        Hidden
        Scene
          │
          │ (indirect paths)
          ▼
    ┌───────────┐
    │   Relay   │◄──── Laser
    │   Wall    │
    └───────────┘
          │
          │ (confocal measurement)
          ▼
       Sensor
```

### 9. The Illuminated Point as a Light Source

Standard path tracing samples the emitter (laser) directly. For NLOS scenes, this is inefficient because:
1. The laser spot is very small
2. Most paths from the hidden scene miss the laser spot entirely
3. We need many samples to find paths that connect through the relay wall

**Better approach:** Treat the illuminated point on the relay wall as a secondary light source.

**Implementation strategy:**

```python
class ConfocalTransientIntegrator(mi.SamplingIntegrator):
    """
    Transient integrator for confocal NLOS rendering.

    Instead of sampling the laser directly, we:
    1. Determine where the laser illuminates the relay wall
    2. Treat that illuminated point as an emitter for NEE
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)
        self.pulse = GaussianPulse(width_opl=props.get('pulse_width_opl', 0.03))

        # Laser parameters
        self.laser_origin = props.get('laser_origin', mi.Point3f(0, 0, 2))
        self.laser_direction = props.get('laser_direction', mi.Vector3f(0, 0, -1))

    def get_laser_hit_point(self, scene) -> mi.SurfaceInteraction3f:
        """Trace the laser ray to find where it hits the relay wall."""
        laser_ray = mi.Ray3f(self.laser_origin, self.laser_direction)
        return scene.ray_intersect(laser_ray)

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        # Get the laser illumination point (once per pixel, or cached)
        si_laser = self.get_laser_hit_point(scene)
        laser_hit_p = si_laser.p
        laser_hit_n = si_laser.n

        # Initialize path tracing
        throughput = mi.Color3f(1.0)
        path_length = mi.Float(0.0)
        # ...

        while dr.hint(active, max_iterations=self.max_depth, label="NLOS"):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            path_length[active] += si.t

            # NEE: Sample the illuminated point instead of the laser
            # Direction from current point to laser hit
            d_to_laser = laser_hit_p - si.p
            dist_to_laser = dr.norm(d_to_laser)
            wo_to_laser = d_to_laser / dist_to_laser

            # Check visibility to the laser hit point
            shadow_ray = si.spawn_ray_to(laser_hit_p)
            visible = ~scene.ray_test(shadow_ray, active)

            # Evaluate BSDF at current point
            # ...

            # The "emitter" contribution from the illuminated point
            # This depends on the laser intensity and relay wall BSDF
            # ...

        return result, mi.Bool(True), []
```

### 10. Point Light vs Area Light Approximation

The illuminated spot on the relay wall can be modeled as:

**Point light approximation (simpler):**
- Treat the laser spot as a single point
- Good when the spot size << scene dimensions

```python
# Point light at laser_hit_p with intensity I
# Contribution to surface point p:
d = laser_hit_p - p
dist = dr.norm(d)
direction = d / dist
contribution = I * bsdf.eval(wo) * dr.dot(n, direction) / (dist * dist)
```

**Area light approximation (more accurate):**
- Model the laser spot as a small area emitter with Gaussian intensity profile
- Sample points on the illuminated area using importance sampling

For the area light implementation, refer to the `ConfocalProjector` class in mitransient:
- [mitransient/emitters/confocal_projector.py](https://github.com/benattal/mitransient/blob/main/mitransient/emitters/confocal_projector.py)

Key methods to study:
- `sample_spot()`: Importance samples positions from a Gaussian mixture using Box-Muller transform
- `eval_pattern()`: Evaluates the Gaussian intensity at a given position
- `sample_emitter()`: Combines spot sampling with emitter direction sampling

The implementation uses:
1. **CDF-based spot selection** when multiple spots exist
2. **Box-Muller transform** for Gaussian sampling: `r = σ × √(-2 log(u₁))`, `θ = 2π × u₂`
3. **Proper PDF computation** combining spot selection probability with Gaussian PDF

### 11. Filtering Direct Visibility

For NLOS rendering, we often want to exclude directly visible contributions (line-of-sight paths). Only include paths that bounce off hidden geometry:

```python
def is_on_relay_wall(si, relay_wall_shapes):
    """Check if intersection is on the relay wall."""
    # Compare shape pointer or use a tag/ID
    return si.shape in relay_wall_shapes

# In the path tracing loop:
# After first intersection, check if we're on the relay wall
on_relay = is_on_relay_wall(si, relay_wall_shapes)

# Only count contributions that go through hidden geometry
# (i.e., paths that leave and return to the relay wall)
include_contribution = had_non_relay_bounce & on_relay
```

### 12. NLOS Scene Setup

Create an NLOS scene with hidden geometry:

```python
nlos_scene = {
    'type': 'scene',

    # Relay wall (visible surface)
    'relay_wall': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([0, 0, 0]),
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}}
    },

    # Hidden geometry (behind an occluder or around a corner)
    'hidden_object': {
        'type': 'sphere',
        'center': [0, 0, -2],  # Behind the relay wall
        'radius': 0.5,
        'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.9, 0.1, 0.1]}}
    },

    # Occluder (blocks direct view of hidden object)
    'occluder': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f.translate([0, 0, -0.5]).rotate([1, 0, 0], 90),
        'bsdf': {'type': 'diffuse'}
    },

    # Transient sensor looking at the relay wall
    'sensor': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at([0, 0, 3], [0, 0, 0], [0, 1, 0]),
        'film': {
            'type': 'transient_hdr_film',
            'width': 128,
            'height': 128,
            'temporal_bins': 256,
            'bin_width_opl': 0.02,
            'start_opl': 4.0,
        }
    }
}
```

### Part 2 Deliverables

1. **Confocal integrator** that treats the laser-illuminated point as a light source
2. **Point light vs area light comparison:** Implement both approximations and compare results
3. **NLOS scene renders:**
   - Transient measurement of hidden geometry
   - Comparison with/without direct visibility filtering
4. **Analysis:**
   - How does spot size affect the rendering?
   - What time bins correspond to different path types (relay-only vs hidden-scene bounce)?

---

## Extensions (Optional)

- **FSCW Rendering:** Implement frequency-swept continuous-wave rendering and compare with the Fourier transform of your transient histogram
- **Depth from AMCW:** Use phase measurements to compute a depth map
- **Time-gated rendering:** Efficiently render only contributions within a specific time window
- **Hidden geometry sampling:** Importance sample directions toward hidden geometry for faster convergence
- **Confocal scanning pattern:** Render a full confocal scan by varying the laser position
