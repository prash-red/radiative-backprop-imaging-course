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

**Render at time function:** This function generates camera rays in a fully vectorized manner and invokes the integrator's `sample` method for a specific target time:

```python
def render_at_time(scene, integrator, sensor, target_time, spp=64):
    """
    Render a single time slice by generating rays and calling the integrator's sample method.

    Args:
        scene: Mitsuba scene
        integrator: TimeGatedTransientPath integrator instance
        sensor: Camera sensor
        target_time: Target optical path length (in meters)
        spp: Samples per pixel

    Returns:
        Rendered image as numpy array with shape (height, width, 3)
    """
    # Get image resolution from sensor
    film = sensor.film()
    res = film.size()
    width, height = res[0], res[1]
    num_pixels = width * height

    # Total number of rays = pixels * spp (fully vectorized)
    total_rays = num_pixels * spp

    # Create and seed sampler for all rays at once
    sampler = mi.load_dict({'type': 'independent', 'sample_count': spp})
    sampler.seed(0, total_rays)

    # VECTORIZED: Create all ray indices at once (including spp dimension)
    idx = dr.arange(mi.UInt32, total_rays)
    pixel_idx = idx // spp  # Which pixel this ray belongs to
    x = pixel_idx % width
    y = pixel_idx // width

    # Convert to normalized coordinates with jitter for anti-aliasing
    jitter = sampler.next_2d()
    pos_x = (mi.Float(x) + jitter.x) / float(width)
    pos_y = (mi.Float(y) + jitter.y) / float(height)
    pos_sample = mi.Point2f(pos_x, pos_y)

    # Samples for ray generation
    time_sample = mi.Float(0.0)
    wavelength_sample = mi.Float(0.5)
    aperture_sample = mi.Point2f(0.5, 0.5)

    # VECTORIZED: Sample all rays at once
    rays, ray_weight = sensor.sample_ray(
        time_sample, wavelength_sample, pos_sample, aperture_sample
    )

    # Call the integrator's sample method with target_time
    # Returns (color, valid, aovs)
    color, valid, _ = integrator.sample(
        scene, sampler, rays, target_time, medium=None, active=True
    )

    # Weight the result
    weighted_color = dr.select(valid, color * ray_weight, mi.Color3f(0.0))

    # Reshape to (num_pixels, spp) and average over spp dimension
    r = np.array(weighted_color.x).reshape(num_pixels, spp).mean(axis=1).reshape(height, width)
    g = np.array(weighted_color.y).reshape(num_pixels, spp).mean(axis=1).reshape(height, width)
    b = np.array(weighted_color.z).reshape(num_pixels, spp).mean(axis=1).reshape(height, width)
    image = np.stack([r, g, b], axis=-1).astype(np.float32)

    return image
```

**Render loop:** To render a full transient image, call `render_at_time()` for many different `target_time` values spanning your temporal range:

```python
def render_transient(scene, integrator, sensor, film_config, spp=64):
    """Render transient by sampling different target times."""
    num_bins = film_config['temporal_bins']
    start_opl = film_config['start_opl']
    bin_width = film_config['bin_width_opl']

    transient_data = []

    for bin_idx in range(num_bins):
        print(f"Rendering bin {bin_idx + 1}/{num_bins}...", end='\r')
        # Target time for this bin (center of bin)
        target_time = start_opl + (bin_idx + 0.5) * bin_width

        # Render at this target time
        image = render_at_time(scene, integrator, sensor, target_time, spp)
        transient_data.append(image)

    print()  # New line after progress
    return np.stack(transient_data, axis=-1)
```

### 4. Test Scene: Transient Cornell Box

Use the provided Cornell Box scene for testing. The scene is adapted from [mitransient](https://github.com/benattal/mitransient/blob/main/mitransient/utils.py):

```python
from scenes import cornell_box, cornell_box_steady_state

# Load the transient Cornell Box scene
scene_dict = cornell_box()
scene = mi.load_dict(scene_dict)

# Or use the steady-state version to test your path tracer first
scene_dict_steady = cornell_box_steady_state()
scene_steady = mi.load_dict(scene_dict_steady)
```

**Rendering both steady state and transient scenes:**

```python
import matplotlib.pyplot as plt

# Configuration
spp = 64
film_config = {
    'temporal_bins': 300,
    'start_opl': 3.5,
    'bin_width_opl': 0.02,
}

# --- Steady State Rendering ---
# Use mi.render for standard steady-state rendering
steady_image = mi.render(scene_steady, spp=spp)
steady_image_np = np.array(steady_image)

# Display steady state result
plt.figure(figsize=(6, 6))
plt.imshow(np.clip(steady_image_np ** (1/2.2), 0, 1))  # Gamma correction
plt.title('Steady State Render')
plt.axis('off')
plt.show()

# --- Transient Rendering ---
# Create the time-gated transient integrator
integrator = TimeGatedTransientPath(mi.Properties())

# Get the sensor from the scene
sensor = scene.sensors()[0]

# Render the full transient using render_transient
transient_data = render_transient(scene, integrator, sensor, film_config, spp=spp)

print(f"Transient data shape: {transient_data.shape}")  # (height, width, 3, temporal_bins)

# Verify: sum over time should approximate steady state
transient_sum = np.sum(transient_data, axis=-1) * film_config['bin_width_opl']
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.clip(steady_image_np ** (1/2.2), 0, 1))
plt.title('Steady State')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np.clip(transient_sum ** (1/2.2), 0, 1))
plt.title('Transient Sum (should match steady state)')
plt.axis('off')
plt.tight_layout()
plt.show()
```

The transient scene is configured with:
- **Film**: 256×256 pixels, 300 temporal bins
- **Temporal range**: starts at OPL 3.5m, bin width 0.02m
- **Integrator**: `transient_path` (from mitransient)

### 5. Visualization

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

### 6. AMCW Rendering

Implement amplitude-modulated continuous-wave (AMCW) ToF rendering. In AMCW systems, the light source and sensor are modulated at frequency ω, and the measurement encodes phase information.

**Background:** Unlike pulsed ToF systems that measure time-of-flight directly, AMCW systems emit continuously modulated light and measure the phase shift between emitted and received signals. The phase shift encodes depth: `depth = (c × phase) / (2 × ω)`.

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

**AMCW Pulse Template:**

```python
class AMCWPulse:
    """
    AMCW (Amplitude-Modulated Continuous-Wave) modulation function.

    Models the correlation measurement in AMCW ToF systems:
        m(t) = cos(ω × t + φ)

    where:
        ω = 2π × frequency (angular frequency)
        φ = phase_offset (measurement phase)
        t = time (converted from OPL as t = opl / c)

    By taking measurements at different phase offsets (e.g., 0° and 90°),
    we can recover the full phasor (amplitude and phase) of the return signal.
    """

    def __init__(self, frequency_hz: float, phase_offset: float = 0.0):
        """
        Initialize the AMCW modulation.

        Args:
            frequency_hz: Modulation frequency in Hz (e.g., 20e6 for 20 MHz)
            phase_offset: Phase offset in radians (0 for cosine, π/2 for sine)
        """
        self.frequency = frequency_hz
        self.phase_offset = phase_offset
        self.c = 299792458.0  # Speed of light (m/s)
        # TODO: Pre-compute angular frequency
        self.omega = ...

    def eval(self, opl: mi.Float) -> mi.Float:
        """
        Evaluate the AMCW modulation at a given optical path length.

        Args:
            opl: Optical path length in meters

        Returns:
            Modulation value cos(ω × t + φ) where t = opl / c
        """
        # TODO: Convert OPL to time, then evaluate modulation
        # time = opl / c
        # return cos(omega * time + phase_offset)
        pass

    def get_unambiguous_range(self) -> float:
        """
        Return the unambiguous depth range for this frequency.

        Returns:
            Maximum unambiguous depth in meters: c / (2 × frequency)
        """
        return self.c / (2.0 * self.frequency)
```

**Testing your AMCW implementation:**

```python
# Test AMCW modulation
amcw_cos = AMCWPulse(frequency_hz=20e6, phase_offset=0.0)        # Cosine (I channel)
amcw_sin = AMCWPulse(frequency_hz=20e6, phase_offset=np.pi/2)    # Sine (Q channel)

print(f"Unambiguous range: {amcw_cos.get_unambiguous_range():.2f} m")

# Test at known OPL
opl_test = mi.Float(1.5)  # 1.5 meters
print(f"Cosine at 1.5m OPL: {amcw_cos.eval(opl_test)}")
print(f"Sine at 1.5m OPL: {amcw_sin.eval(opl_test)}")
```

**Dual-Phase AMCW Rendering:**

To recover the full phasor (amplitude and phase), render the scene twice with AMCW pulses at 0° and 90° phase offsets:

```python
class AMCWTransientPath(mi.SamplingIntegrator):
    """
    AMCW integrator that weights path contributions by sinusoidal modulation.

    This integrator computes the correlation integral:
        I(φ) = ∫ L(opl) × cos(ω × opl/c + φ) d(opl)

    where L(opl) is the transient radiance at optical path length opl.
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)
        self.amcw = None  # Set externally before rendering

    def set_modulation(self, amcw_pulse: AMCWPulse):
        """Set the AMCW modulation function."""
        self.amcw = amcw_pulse

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        """
        Sample paths and weight by AMCW modulation.

        Unlike pulsed transient rendering, AMCW weights each path contribution
        by the modulation function evaluated at the path's optical path length.
        """
        result = mi.Color3f(0.0)
        throughput = mi.Color3f(1.0)
        ray = mi.Ray3f(ray)
        active = mi.Bool(active)
        path_length = mi.Float(0.0)

        while dr.hint(active, max_iterations=self.max_depth, label="AMCW"):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()

            # Accumulate path length
            path_length[active] += si.t

            # For each light contribution (NEE or emitter hit):
            # total_opl = path_length + distance_to_light
            # amcw_weight = self.amcw.eval(total_opl)
            # result += throughput * radiance * amcw_weight

            # ... your path tracing code ...

        return result, mi.Bool(True), []


def render_amcw_phasor(scene, sensor, frequency_hz, spp=64):
    """
    Render AMCW phasor image by capturing I (cosine) and Q (sine) channels.

    Args:
        scene: Mitsuba scene
        sensor: Camera sensor
        frequency_hz: Modulation frequency in Hz
        spp: Samples per pixel

    Returns:
        Tuple of (I_image, Q_image, amplitude, phase, depth) where:
        - I_image: Cosine correlation (real part)
        - Q_image: Sine correlation (imaginary part)
        - amplitude: sqrt(I² + Q²)
        - phase: atan2(Q, I)
        - depth: phase converted to depth
    """
    # Create integrator
    integrator = AMCWTransientPath(mi.Properties())

    # --- Render I channel (cosine, phase = 0) ---
    amcw_cos = AMCWPulse(frequency_hz=frequency_hz, phase_offset=0.0)
    integrator.set_modulation(amcw_cos)
    I_image = render_single_amcw(scene, integrator, sensor, spp)

    # --- Render Q channel (sine, phase = π/2) ---
    amcw_sin = AMCWPulse(frequency_hz=frequency_hz, phase_offset=np.pi/2)
    integrator.set_modulation(amcw_sin)
    Q_image = render_single_amcw(scene, integrator, sensor, spp)

    # --- Compute amplitude and phase ---
    amplitude = np.sqrt(I_image**2 + Q_image**2)
    phase = np.arctan2(Q_image, I_image)

    # Convert phase to depth (unwrapped within unambiguous range)
    c = 299792458.0
    omega = 2 * np.pi * frequency_hz
    depth = (c * phase) / (2 * omega)

    return I_image, Q_image, amplitude, phase, depth


def render_single_amcw(scene, integrator, sensor, spp):
    """Render a single AMCW channel using vectorized ray generation."""
    film = sensor.film()
    res = film.size()
    width, height = res[0], res[1]
    num_pixels = width * height
    total_rays = num_pixels * spp

    sampler = mi.load_dict({'type': 'independent', 'sample_count': spp})
    sampler.seed(0, total_rays)

    idx = dr.arange(mi.UInt32, total_rays)
    pixel_idx = idx // spp
    x = pixel_idx % width
    y = pixel_idx // width

    jitter = sampler.next_2d()
    pos_x = (mi.Float(x) + jitter.x) / float(width)
    pos_y = (mi.Float(y) + jitter.y) / float(height)
    pos_sample = mi.Point2f(pos_x, pos_y)

    time_sample = mi.Float(0.0)
    wavelength_sample = mi.Float(0.5)
    aperture_sample = mi.Point2f(0.5, 0.5)

    rays, ray_weight = sensor.sample_ray(
        time_sample, wavelength_sample, pos_sample, aperture_sample
    )

    color, valid, _ = integrator.sample(
        scene, sampler, rays, medium=None, active=True
    )

    weighted_color = dr.select(valid, color * ray_weight, mi.Color3f(0.0))

    # Average over spp and reshape
    r = np.array(weighted_color.x).reshape(num_pixels, spp).mean(axis=1).reshape(height, width)
    g = np.array(weighted_color.y).reshape(num_pixels, spp).mean(axis=1).reshape(height, width)
    b = np.array(weighted_color.z).reshape(num_pixels, spp).mean(axis=1).reshape(height, width)

    return np.stack([r, g, b], axis=-1).astype(np.float32)
```

**Visualizing AMCW Results:**

```python
# Render AMCW phasor at 20 MHz
I, Q, amplitude, phase, depth = render_amcw_phasor(scene, sensor, frequency_hz=20e6, spp=64)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# I channel (cosine correlation)
axes[0, 0].imshow(I.mean(axis=-1), cmap='RdBu')
axes[0, 0].set_title('I Channel (Cosine)')
axes[0, 0].axis('off')

# Q channel (sine correlation)
axes[0, 1].imshow(Q.mean(axis=-1), cmap='RdBu')
axes[0, 1].set_title('Q Channel (Sine)')
axes[0, 1].axis('off')

# Amplitude
axes[0, 2].imshow(amplitude.mean(axis=-1), cmap='viridis')
axes[0, 2].set_title('Amplitude')
axes[0, 2].axis('off')

# Phase
axes[1, 0].imshow(phase.mean(axis=-1), cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[1, 0].set_title('Phase')
axes[1, 0].axis('off')

# Depth (from phase)
axes[1, 1].imshow(depth.mean(axis=-1), cmap='plasma')
axes[1, 1].set_title('Depth from Phase')
axes[1, 1].axis('off')

# Compare with different frequency (higher resolution, shorter range)
I2, Q2, amp2, phase2, depth2 = render_amcw_phasor(scene, sensor, frequency_hz=50e6, spp=64)
axes[1, 2].imshow(depth2.mean(axis=-1), cmap='plasma')
axes[1, 2].set_title('Depth at 50 MHz')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Print unambiguous ranges
print(f"20 MHz unambiguous range: {299792458 / (2 * 20e6):.2f} m")
print(f"50 MHz unambiguous range: {299792458 / (2 * 50e6):.2f} m")
```

**Implementation task:** Create an AMCW integrator that outputs phasor images (real and imaginary channels).

### Part 1 Deliverables

1. **GaussianPulse implementation** with `eval()` and `sample()` methods
2. **TimeGatedTransientPath integrator** that takes `target_time` as input
3. **Visualizations:**
   - Transient animation showing light propagation
   - Single-pixel temporal responses at different scene locations
   - Steady-state image (sum over all time bins)
4. **AMCW phasor images** at 2-3 different modulation frequencies

---

## Project Part 2: TransientHDRFilm and More Efficient Transient Integrators

In this part, you will work with the `TransientHDRFilm` from mitransient and build more efficient transient integrators. In particular, for these transient integrators, every path vertex will contribute to the final measurement.

### 7. Understanding the TransientHDRFilm

The `TransientHDRFilm` from mitransient stores measurements in time bins. The Cornell Box scene already includes a properly configured transient film:

```python
'film': {
    'type': 'transient_hdr_film',
    'width': 256,
    'height': 256,
    'temporal_bins': 300,       # Number of time bins
    'bin_width_opl': 0.02,      # Width per bin in meters
    'start_opl': 3.5,           # Starting optical path length
}
```

The film provides methods to:
- `prepare()`: Initialize the film for rendering
- `develop()`: Return `(steady_image, transient_data)` where transient_data has shape `(height, width, temporal_bins, channels)`

### 8. Building a Transient Path Tracer

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

### 9. Rendering with Different Light Source Profiles

Use your `TransientPath` integrator to render transient measurements with different light source profiles, replicating the results from Part 1.

**Task 1: Render with a Gaussian Pulse**

Render the Cornell Box scene using your `TransientPath` integrator with a `GaussianPulse`. Compare the results with your `TimeGatedTransientPath` from Part 1.

```python
# Render transient with Gaussian pulse using TransientPath
integrator = TransientPath(mi.Properties())
integrator.pulse = GaussianPulse(width_opl=0.03)

# Render and develop the transient film
# transient_data shape: (height, width, temporal_bins, channels)
steady_image, transient_data = film.develop()

# Visualize using the functions from Section 5
visualize_transient(transient_data, start_opl=3.5, bin_width_opl=0.02)
```

**Task 2: Render with an AMCW Pulse**

Now render using your `AMCWPulse` class. Since the `TransientPath` integrator outputs time-binned transient data, you need to **convert the transient measurements to phasor measurements** in post-processing.

The key insight is that AMCW measurements are a weighted integral of the transient response:

```
I(φ) = ∫ L(t) × cos(ωt + φ) dt
```

This can be computed from discrete transient bins as:

```python
def transient_to_phasor(transient_data, frequency_hz, start_opl, bin_width_opl):
    """
    Convert transient measurements to AMCW phasor measurements.

    Args:
        transient_data: Array of shape (height, width, temporal_bins, channels)
        frequency_hz: AMCW modulation frequency in Hz
        start_opl: Starting optical path length (meters)
        bin_width_opl: Width of each time bin (meters)

    Returns:
        Tuple of (I_channel, Q_channel, amplitude, phase, depth)
    """
    c = 299792458.0  # Speed of light (m/s)
    omega = 2 * np.pi * frequency_hz
    num_bins = transient_data.shape[2]

    # Compute OPL for each bin center
    bin_centers = start_opl + (np.arange(num_bins) + 0.5) * bin_width_opl

    # Convert OPL to time and compute modulation weights
    times = bin_centers / c
    cos_weights = np.cos(omega * times)  # Shape: (num_bins,)
    sin_weights = np.sin(omega * times)  # Shape: (num_bins,)

    # Integrate transient with modulation (sum over time bins)
    # transient_data: (H, W, T, C), weights: (T,) -> broadcast and sum
    I_channel = np.sum(transient_data * cos_weights[None, None, :, None], axis=2) * bin_width_opl
    Q_channel = np.sum(transient_data * sin_weights[None, None, :, None], axis=2) * bin_width_opl

    # Compute amplitude and phase
    amplitude = np.sqrt(I_channel**2 + Q_channel**2)
    phase = np.arctan2(Q_channel, I_channel)

    # Convert phase to depth
    depth = (c * phase) / (2 * omega)

    return I_channel, Q_channel, amplitude, phase, depth


# Example usage:
I, Q, amplitude, phase, depth = transient_to_phasor(
    transient_data,
    frequency_hz=20e6,
    start_opl=3.5,
    bin_width_opl=0.02
)

# Compare with your Part 1 AMCW results
# The phasor images should match!
```

**Task 3: Compare Results**

Verify that your transient-to-phasor conversion produces the similar results as directly rendering with `AMCWTransientPath` from Part 1:

```python
# Render at multiple frequencies and compare
for freq in [20e6, 50e6]:
    # Method 1: Direct AMCW rendering (from Part 1)
    I_direct, Q_direct, _, _, _ = render_amcw_phasor(scene, sensor, freq, spp=64)

    # Method 2: Convert transient to phasor
    I_conv, Q_conv, _, _, _ = transient_to_phasor(transient_data, freq, start_opl, bin_width_opl)

    # Compare (should be very similar)
    print(f"Frequency: {freq/1e6:.0f} MHz")
    print(f"  I channel MSE: {np.mean((I_direct - I_conv)**2):.6f}")
    print(f"  Q channel MSE: {np.mean((Q_direct - Q_conv)**2):.6f}")
```

### Part 2 Deliverables

1. **TransientPath integrator** that renders to a `TransientHDRFilm`
2. **Gaussian pulse transient render** using `TransientPath`
3. **AMCW phasor images** computed from transient data using `transient_to_phasor()`
4. **Comparison** showing that direct AMCW rendering and transient-to-phasor conversion produce similar results

---

## Project Part 3: Confocal Scanning and NLOS Rendering

In this part, you will implement efficient rendering for confocal scanning systems and non-line-of-sight (NLOS) imaging. The key insight is that instead of sampling light sources directly, we can treat the laser-illuminated point on the relay wall as a secondary light source.

### 10. Confocal Scanning Setup

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

### 11. The Illuminated Point as a Light Source

Standard path tracing samples the emitter (laser) directly. For NLOS scenes, this is inefficient because:
1. The laser spot is very small
2. Most paths from the hidden scene miss the laser spot entirely
3. We need many samples to find paths that connect through the relay wall

**Better approach:** Treat the illuminated point on the relay wall as a secondary light source.

**Key insight for confocal scanning:** In a confocal setup, the laser direction is aligned with the camera ray direction. This means the laser illuminates the same point on the relay wall that the camera pixel is observing. For each camera ray, trace it to find the relay wall intersection—this is also where the laser hits.

**Implementation strategy:**

```python
class ConfocalTransientIntegrator(mi.SamplingIntegrator):
    """
    Transient integrator for confocal NLOS rendering.

    In confocal mode, the laser is co-located with the camera and points
    in the same direction as each camera ray. This means:
    1. The first intersection of the camera ray IS the laser hit point
    2. We treat this illuminated point as an emitter for subsequent bounces
    """

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = props.get('max_depth', 8)
        self.pulse = GaussianPulse(width_opl=props.get('pulse_width_opl', 0.03))

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        # First intersection: this is the confocal point (laser hits here too)
        si_confocal = scene.ray_intersect(ray, active)
        confocal_p = si_confocal.p
        confocal_n = si_confocal.n

        # Path length starts with distance to confocal point
        # (light travels: laser -> confocal point -> hidden scene -> confocal point -> sensor)
        path_length = mi.Float(si_confocal.t)

        # Now trace secondary rays from the confocal point into the scene
        # Sample a direction from the confocal point's hemisphere
        wo_local = mi.warp.square_to_cosine_hemisphere(sampler.next_2d())
        wo_world = si_confocal.to_world(wo_local)
        secondary_ray = si_confocal.spawn_ray(wo_world)

        # Continue path tracing from the secondary ray
        # ...

        while dr.hint(active, max_iterations=self.max_depth, label="NLOS"):
            si = scene.ray_intersect(secondary_ray, active)
            active &= si.is_valid()
            path_length[active] += si.t

            # NEE: Sample the confocal point as a light source
            d_to_confocal = confocal_p - si.p
            dist_to_confocal = dr.norm(d_to_confocal)
            wo_to_confocal = d_to_confocal / dist_to_confocal

            # Check visibility back to the confocal point
            shadow_ray = si.spawn_ray_to(confocal_p)
            visible = ~scene.ray_test(shadow_ray, active)

            # The contribution includes the return path to confocal point
            # total_path = path_length + dist_to_confocal + dist_to_confocal (return to sensor)
            # ...

        return result, mi.Bool(True), []
```

### 12. Point Light vs Area Light Approximation

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

**Area light approximation (more accurate, optional):**
- Model the laser spot as a small area emitter with Gaussian intensity profile
- Sample rays from the projector and trace them to find illuminated points

For the area light implementation, refer to the `ConfocalProjector` class in mitransient:
- [mitransient/emitters/confocal_projector.py](https://github.com/benattal/mitransient/blob/main/mitransient/emitters/confocal_projector.py)

Key methods to study:
- `sample_spot()`: Samples *rays from the projector* using a Gaussian distribution (Box-Muller transform)
- `eval_pattern()`: Evaluates the Gaussian intensity at a given position
- `sample_emitter()`: Combines spot sampling with emitter direction sampling

**Note:** The `sample_spot()` method samples in the projector's ray space, not directly on the illuminated surface. Converting this to a valid PDF over illuminated points requires accounting for the Jacobian of the projection. This is a subtle detail that may be beyond the scope of this assignment—the point light approximation is sufficient for the core deliverables.

### 13. Filtering Direct Visibility

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

### 14. NLOS Scene Setup

For NLOS testing, use the example scenes from mitransient. The repository includes several NLOS configurations:

- `wall_box_sphere_nlos.xml` - Relay wall with hidden sphere and cube
- `wall_box_spheres_nlos.xml` - Relay wall with multiple hidden spheres
- `ourbox_confocal.xml` - Confocal scanning configuration

You can download these from the [mitransient scenes directory](https://github.com/benattal/mitransient/tree/main/scenes):

```python
# Load an NLOS scene from XML
scene = mi.load_file('path/to/wall_box_sphere_nlos.xml')

# Or define a simple NLOS scene programmatically
# See mitransient/scenes/ for complete examples
```

The NLOS scenes typically include:
- **Relay wall**: A diffuse surface that the sensor observes
- **Hidden geometry**: Objects behind/around the relay wall (not directly visible)
- **Confocal projector**: Laser that illuminates points on the relay wall
- **Transient film**: Configured to capture the time-resolved response

### Part 3 Deliverables

1. **Confocal integrator** that treats the laser-illuminated point as a point light source
2. **NLOS scene renders:**
   - Transient measurement of hidden geometry
   - Comparison with/without direct visibility filtering
3. **Analysis:**
   - What time bins correspond to different path types (relay-only vs hidden-scene bounce)?
   - How does the transient response reveal information about the hidden geometry?

**Optional:** Implement the area light approximation (see `ConfocalProjector` in mitransient) and compare with the point light approach.