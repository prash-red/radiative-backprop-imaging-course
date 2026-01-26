import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

class DirectIntegrator(mi.SamplingIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

    def sample(self, scene, sampler, ray, medium=None, active=True):
        # Intersect ray with scene
        si = scene.ray_intersect(ray, active)

        # Return black if no intersection
        result = mi.Color3f(0.0)
        active = si.is_valid()

        if not dr.any(active):
            return result, active, []

        # Get emitter value if we hit a light source
        result[active] = si.emitter(scene).eval(si, active)

        # Sample direct illumination from light sources
        ctx = mi.BSDFContext()
        bsdf = si.bsdf()

        # Sample a light source
        ds, emitter_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active
        )

        active_light = active & (ds.pdf > 0)

        if dr.any(active_light):
            # Evaluate BSDF
            bsdf_val = bsdf.eval(ctx, si, si.to_local(ds.d), active_light)

            # Visibility test
            ray_shadow = si.spawn_ray_to(ds.p)
            visible = ~scene.ray_test(ray_shadow, active_light)

            # Accumulate contribution
            result[active_light & visible] += (
                bsdf_val * emitter_weight * dr.abs(dr.dot(si.n, ds.d))
            )

        return result, active, []

class SimpleIntegrator(mi.SamplingIntegrator):
    def __init__(self, sample_lights = True, sample_bsdf=True, max_depth=5, props=mi.Properties()):
        super().__init__(props)
        self.sample_lights = sample_lights
        self.sample_bsdf = sample_bsdf
        self.max_depth = max_depth

    def sample(self, scene, sampler, ray, medium=None, active=True):
        # Intersect ray with scene
        beta = mi.Color3f(1.0)
        result = mi.Color3f(0.0)
        specular = dr.auto.ad.Bool(True)
        
        for depth in range(self.max_depth):
            si = scene.ray_intersect(ray, active)

            active &= si.is_valid()

            if not dr.any(active):
                # Add environment contribution for rays that didn't hit anything
                env_mask = active & (specular | ~dr.auto.ad.Bool(self.sample_lights))
                result[env_mask] += beta * scene.environment().eval(si, env_mask)
                break

            # Add emitter contribution if we hit a light and either sampling specular or not sampling lights
            emitter_mask = active & (specular | ~dr.auto.ad.Bool(self.sample_lights))
            result[emitter_mask] += beta * si.emitter(scene).eval(si, emitter_mask)

            # Sample direct illumination from light sources
            ctx = mi.BSDFContext()
            bsdf = si.bsdf()

            if self.sample_lights:
                # Sample a light source
                ds, emitter_weight = scene.sample_emitter_direction(
                    si, sampler.next_2d(), True, active
                )

                active_light = active & (ds.pdf > 0)

                if dr.any(active_light):
                    # Evaluate BSDF
                    bsdf_val = bsdf.eval(ctx, si, si.to_local(ds.d), active_light)

                    # Visibility test
                    ray_shadow = si.spawn_ray_to(ds.p)
                    visible = ~scene.ray_test(ray_shadow, active_light)

                    # Accumulate contribution
                    result[active_light & visible] += (
                        beta * bsdf_val * emitter_weight
                    )
            
            # Sample BSDF to get new direction
            if (self.sample_bsdf):
                bsdf_sample, bsdf_weight = bsdf.sample(
                    ctx, si, sampler.next_1d(), sampler.next_2d(), active
                )

                active_bsdf = active & dr.any(bsdf_weight != 0)

                # Update beta with mask
                beta[active_bsdf] = beta * bsdf_weight

                # Spawn new ray with mask
                ray[active_bsdf] = si.spawn_ray(si.to_world(bsdf_sample.wo))

                # Update specular flag with mask
                specular[active_bsdf] = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
                
                # Update active mask - continue only where BSDF sampling succeeded
                active &= active_bsdf
            else:
                # uniformly sample hemisphere
                wi = sampler.next_2d()
                local_dir = mi.warp.square_to_uniform_hemisphere(wi)
                bsdf_val = bsdf.eval(ctx, si, local_dir, active)
                pdf = mi.warp.square_to_uniform_hemisphere_pdf(local_dir)
                bsdf_weight = bsdf_val * dr.abs(dr.dot(si.n, si.to_world(local_dir))) / pdf
                
                # Update with mask
                beta[active] = beta * bsdf_weight
                ray[active] = si.spawn_ray(si.to_world(local_dir))
                specular[active] = dr.auto.Bool(False)

        return result, active, []

def power_heuristic(pdf_a, pdf_b, beta=2):
    a = dr.power(pdf_a, beta)
    b = dr.power(pdf_b, beta)
    return dr.select(pdf_a > 0, a / (a + b), 0)

class MISIntegrator(mi.SamplingIntegrator):
    def __init__(self, max_depth=5, props=mi.Properties()):
        super().__init__(props)
        self.max_depth = max_depth

    def sample(self, scene, sampler, ray, medium=None, active=True):
        # Intersect ray with scene
        beta = mi.Color3f(1.0)
        result = mi.Color3f(0.0)
        specular = dr.auto.ad.Bool(False)
        
        for depth in range(self.max_depth): 
            si = scene.ray_intersect(ray, active)

            active &= si.is_valid()

            if not dr.any(active):
                # Add environment contribution for rays that didn't hit anything
                env_mask = not active & (specular) 
                result[env_mask] += beta * scene.environment().eval(si, env_mask)
                break

            # Add emitter contribution if we hit a light and either sampling specular or not sampling lights
            emitter_mask = active & (specular | depth == 0) 
            result[emitter_mask] += beta * si.emitter(scene).eval(si, emitter_mask)
            

            # Sample direct illumination from light sources
            ctx = mi.BSDFContext()
            bsdf = si.bsdf()

            # Sample a light source
            ds, emitter_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active
            )
            
            bsdf_val = bsdf.eval(ctx, si, si.to_local(ds.d), active)
            bsdf_pdf = bsdf.pdf(ctx, si, si.to_local(ds.d), active)
            mis_weight_light = power_heuristic(ds.pdf, bsdf_pdf)
            visible = ~scene.ray_test(si.spawn_ray_to(ds.p), active)
            result[visible] += mis_weight_light * beta * emitter_weight * bsdf_val * dr.abs(dr.dot(si.n, ds.d)) 
            
            bsdf_sample, bsdf_weight = bsdf.sample(
                ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )
            ray_bsdf = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si_bsdf = scene.ray_intersect(ray_bsdf, active)
            hit_emitter = active & si_bsdf.is_valid() & (si_bsdf.emitter(scene) != None)

            if dr.any(hit_emitter):
                Le = si_bsdf.emitter(scene).eval(si_bsdf, hit_emitter)
                ds_bsdf = mi.DirectionSample3f(scene, si_bsdf, si)
                light_pdf = scene.pdf_emitter_direction(si, ds_bsdf, hit_emitter)
                mis_weight_bsdf = power_heuristic(bsdf_sample.pdf, light_pdf)
                result[hit_emitter] += mis_weight_bsdf * beta * bsdf_weight * Le

            active_bsdf = active & dr.any(bsdf_weight != 0)

            # Update beta with mask
            beta[active_bsdf] = beta * bsdf_weight

            # Spawn new ray with mask
            ray[active_bsdf] = ray_bsdf 

            specular[active_bsdf] = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            
            # Update active mask - continue only where BSDF sampling succeeded
            active &= active_bsdf


        return result, active, []



# Register the integrator
mi.register_integrator("direct", lambda props: DirectIntegrator(props))
mi.register_integrator("simple", lambda props: SimpleIntegrator(
    props.get_bool("sample_lights", True),
    props.get_bool("sample_bsdf", True),
    props.get_int("max_depth", 5),
    props
))
mi.register_integrator("mis", lambda props: MISIntegrator(
    props.get_int("max_depth", 5),
    props
))

# integrator = SimpleIntegrator(sample_bsdf=True, sample_lights=True, max_depth=5)
integrator = MISIntegrator(max_depth=10)

# Load and render the scene
scene = mi.load_file('cornell_box.xml')
image = mi.render(scene,integrator=integrator, spp=124)
path_traced_image =  mi.render(scene, spp=124)

mi.util.write_bitmap('direct_integrator_render.png', image)
mi.util.write_bitmap('path_traced_render.png', path_traced_image)
