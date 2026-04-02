If all went well, this directory is empty; otherwise it contains the last forward model evaluation 
before the geoVi sampling algorithm returned a nan energy error. Unpickle the pickle file stored 
in this file and analyze it by applying the forward model to the latent positions.

In order for this functionality to work, currently, the following lines need to be added to the 
function `residual_vg` defined inside `nifty.re.evi.nonlinearly_update_residual`:
        import datetime
        import jax
        import pickle
        current_date = datetime.datetime.now()
        fn = 'STRAIN_GW150914_H1_osc_dlt_later/errors/nan_state_'+str(current_date)+'.pkl'
        def save_nan_state(position, energy, filename=fn):
            if jnp.isnan(energy):
                pos_host = jax.tree_util.tree_map(jax.device_get, position)
                with open(filename, "wb") as f:
                    pickle.dump(pos_host, f)
                print(f"Saved NaN state")
        jax.debug.callback(save_nan_state, x, res)
        