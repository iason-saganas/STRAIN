from helpers import *
key = jax.random.PRNGKey(42)

time, strain = get_sample_data()

pipe_1 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key)
pipe_1.add_cfm_signal_model(fluct=(5,2), llslope=(-4,1))
pipe_1.add_noise_op(noise_var_level=1e-10)

latent_samples = pipe_1.run_inference(kl_iterations=5, use_strict_minimizers=False, out_name="re_pipe_1", resume=True)
# key = pipe_1.get_current_key()